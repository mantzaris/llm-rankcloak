#!/usr/bin/env Rscript

# DRAFT analysis script. Fits cumulative-link mixed models with ordinal::clmm.
# It never recruits, submits, pays, or contacts participants.

parse_args <- function(args) {
  result <- list(data = NULL, output_dir = NULL, validate_only = FALSE)
  i <- 1
  while (i <= length(args)) {
    if (args[[i]] == "--data") {
      i <- i + 1; result$data <- args[[i]]
    } else if (args[[i]] == "--output-dir") {
      i <- i + 1; result$output_dir <- args[[i]]
    } else if (args[[i]] == "--validate-only") {
      result$validate_only <- TRUE
    } else {
      stop(paste("unknown argument", args[[i]]))
    }
    i <- i + 1
  }
  if (is.null(result$data)) stop("--data is required")
  if (is.null(result$output_dir)) stop("--output-dir is required")
  result
}

holm_adjust <- function(p) {
  p.adjust(p, method = "holm")
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
data <- read.csv(args$data, stringsAsFactors = FALSE, check.names = FALSE)
required <- c(
  "participant_slot_id", "item_type", "stimulus_blind_id", "scale_id", "rating",
  "condition", "prompt_category", "synthetic_fixture", "include_primary"
)
missing <- setdiff(required, names(data))
if (length(missing)) stop(paste("missing required columns:", paste(missing, collapse = ", ")))

ratings <- data[data$item_type == "experimental_message" & data$include_primary == "true", ]
scales <- c(
  "grammaticality", "fluency", "coherence", "topic_adherence", "completeness",
  "overall_naturalness", "suspiciousness"
)
if (!all(ratings$scale_id %in% scales)) stop("unexpected experimental scale_id")
if (!all(ratings$rating %in% 1:7)) stop("ratings must be integers 1 through 7")
if (length(unique(ratings$condition)) != 8) stop("expected eight conditions")
if (length(unique(ratings$scale_id)) != 7) stop("expected seven scales")

per_stimulus <- table(ratings$stimulus_blind_id, ratings$scale_id)
if (!all(per_stimulus == 3)) stop("each stimulus/scale must have exactly three ratings")
per_participant <- table(ratings$participant_slot_id) / length(scales)
if (!all(per_participant == 24)) stop("each fixture panel slot must have 24 messages")

validation <- data.frame(
  field = c("experimental_rows", "participants", "stimuli", "conditions", "scales"),
  value = c(nrow(ratings), length(unique(ratings$participant_slot_id)),
            length(unique(ratings$stimulus_blind_id)), length(unique(ratings$condition)),
            length(unique(ratings$scale_id)))
)
write.csv(validation, file.path(args$output_dir, "validation_summary.csv"), row.names = FALSE)
if (args$validate_only) {
  cat("schema/design validation passed\n")
  quit(status = 0)
}

if (!requireNamespace("ordinal", quietly = TRUE)) {
  stop("Package 'ordinal' is required for fitting. Install it only in the approved locked analysis environment.")
}

ratings$condition <- relevel(factor(ratings$condition), ref = "ordinary_llm_control")
ratings$prompt_category <- factor(ratings$prompt_category)
ratings$participant_slot_id <- factor(ratings$participant_slot_id)
ratings$stimulus_blind_id <- factor(ratings$stimulus_blind_id)
ratings$rating_ordered <- ordered(ratings$rating, levels = 1:7)

all_coefficients <- list()
for (scale in scales) {
  subset <- ratings[ratings$scale_id == scale, ]
  fit <- ordinal::clmm(
    rating_ordered ~ condition + prompt_category +
      (1 | participant_slot_id) + (1 | stimulus_blind_id),
    data = subset, link = "logit", Hess = TRUE, nAGQ = 1,
    control = ordinal::clmm.control(method = "nlminb", maxIter = 1000)
  )
  capture.output(summary(fit), file = file.path(args$output_dir, paste0(scale, "_summary.txt")))
  coefficients <- coef(summary(fit))
  terms <- rownames(coefficients)
  fixed <- grepl("^condition", terms)
  if (any(fixed)) {
    block <- data.frame(
      scale_id = scale,
      term = terms[fixed],
      estimate = coefficients[fixed, "Estimate"],
      std_error = coefficients[fixed, "Std. Error"],
      z_value = coefficients[fixed, "z value"],
      p_value = coefficients[fixed, "Pr(>|z|)"],
      stringsAsFactors = FALSE
    )
    block$odds_ratio <- exp(block$estimate)
    block$ci95_low <- exp(block$estimate - 1.96 * block$std_error)
    block$ci95_high <- exp(block$estimate + 1.96 * block$std_error)
    all_coefficients[[scale]] <- block
  }
}

results <- do.call(rbind, all_coefficients)
primary <- results$scale_id %in% c("overall_naturalness", "suspiciousness")
results$holm_family <- ifelse(primary, "co_primary_condition_coefficients", "secondary_condition_coefficients")
results$p_holm <- NA_real_
results$p_holm[primary] <- holm_adjust(results$p_value[primary])
results$p_holm[!primary] <- holm_adjust(results$p_value[!primary])
write.csv(results, file.path(args$output_dir, "condition_coefficients.csv"), row.names = FALSE)
cat("fitted", length(scales), "cumulative-link mixed models\n")
