#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE, warn = 1)

PROTOCOL_CONTRACT_REVISION <- "payload_fidelity_v2"
RESULT_SCHEMA_REVISION <- "payload_aware_result_v2"
PAYLOAD_RECOVERY_SEMANTICS <- "original_serialized_payload_bytes_sha256_v1"
PRIMARY_EVIDENCE_STATUS <- "confirmatory_primary_v2_payload_fidelity_after_manifest_freeze"
PRIMARY_STUDY_PHASE <- "primary_v2_confirmatory"

`%||%` <- function(value, fallback) {
  if (is.null(value) || length(value) == 0L) fallback else value
}

abort <- function(message) {
  stop(message, call. = FALSE)
}

parse_cli <- function(arguments) {
  result <- list(validate_only = FALSE)
  index <- 1L
  while (index <= length(arguments)) {
    flag <- arguments[[index]]
    if (flag == "--validate-only") {
      result$validate_only <- TRUE
      index <- index + 1L
      next
    }
    if (!startsWith(flag, "--")) {
      abort(sprintf("Unexpected positional argument: %s", flag))
    }
    if (index == length(arguments)) {
      abort(sprintf("Missing value for %s", flag))
    }
    key <- gsub("-", "_", substring(flag, 3L), fixed = TRUE)
    result[[key]] <- arguments[[index + 1L]]
    index <- index + 2L
  }
  result
}

require_option <- function(options, name) {
  value <- options[[name]]
  if (is.null(value) || !nzchar(value)) {
    abort(sprintf("Required option --%s is missing", gsub("_", "-", name)))
  }
  value
}

read_json <- function(path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    abort("jsonlite is required for plan and manifest I/O")
  }
  if (!file.exists(path)) abort(sprintf("JSON input does not exist: %s", path))
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

write_json <- function(value, path) {
  jsonlite::write_json(
    value,
    path,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null",
    na = "null",
    digits = NA
  )
  cat("\n", file = path, append = TRUE)
}

read_csv_required <- function(path, label) {
  if (!file.exists(path)) abort(sprintf("%s input does not exist: %s", label, path))
  data <- utils::read.csv(
    path,
    check.names = FALSE,
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN")
  )
  if (nrow(data) == 0L) abort(sprintf("%s input is empty", label))
  data
}

read_csv_optional <- function(path, label) {
  if (is.null(path)) return(NULL)
  read_csv_required(path, label)
}

verify_feature_join_manifest <- function(manifest_path, features_path, features) {
  manifest <- read_json(manifest_path)
  if (!identical(manifest$schema_version, "rankcloak-revision-heldout-feature-join-v1") ||
      !identical(manifest$manifest_type, "rankcloak_revision_primary_heldout_feature_join")) {
    abort("Unsupported held-out evaluator feature-join manifest")
  }
  if (!identical(manifest$input_scope, "primary_v2_rankcloak_full_message_only") ||
      !identical(manifest$source_record_hashes_recomputed, TRUE) ||
      !identical(manifest$evaluator_source_records_byte_identical_to_preprocessing, TRUE) ||
      !identical(manifest$evaluator_artifact_pins_verified, TRUE) ||
      !identical(manifest$segments_as_independent_observations, FALSE) ||
      !identical(manifest$protocol_contract_revision, PROTOCOL_CONTRACT_REVISION) ||
      !identical(manifest$result_schema_revision, RESULT_SCHEMA_REVISION) ||
      !identical(manifest$unmatched_primary_trials, 0L) ||
      !identical(manifest$duplicate_evaluator_trial_ids, 0L)) {
    abort("Held-out evaluator feature-join manifest failed its primary contract")
  }
  pins <- manifest$evaluator_artifact_pins
  if (!is.list(pins) || length(pins) != 3L ||
      is.null(manifest$models_config_sha256) ||
      !grepl("^[0-9a-f]{64}$", manifest$models_config_sha256) ||
      any(!vapply(pins, function(value) {
        is.character(value) && length(value) == 1L &&
          grepl("^[0-9a-f]{64}$", value)
      }, logical(1)))) {
    abort("Held-out evaluator feature join lacks the three frozen artifact pins")
  }
  declaration <- manifest$outputs$features
  if (is.null(declaration) || is.null(declaration$path) || is.null(declaration$sha256)) {
    abort("Held-out evaluator feature-join manifest lacks its features identity")
  }
  declared_path <- declaration$path
  candidate <- if (grepl("^/", declared_path)) {
    declared_path
  } else {
    file.path(dirname(manifest_path), declared_path)
  }
  if (!identical(
    normalizePath(candidate, mustWork = TRUE),
    normalizePath(features_path, mustWork = TRUE)
  )) abort("--features is not the table declared by --feature-join-manifest")
  if (!identical(sha256_file(features_path), declaration$sha256)) {
    abort("Joined feature-table SHA-256 differs from its manifest")
  }
  if (!identical(unname(file.info(features_path)$size), as.numeric(declaration$size_bytes)) ||
      !identical(nrow(features), as.integer(declaration$row_count)) ||
      !identical(length(unique(features$trial_id)), as.integer(manifest$primary_trial_count)) ||
      !identical(as.integer(manifest$primary_trial_count), 6480L)) {
    abort("Joined feature-table dimensions differ from the complete primary contract")
  }
  artifact_diagnostics <- manifest$artifact_diagnostics
  expected_artifact_candidates <- c(
    "artifact_count", "surface_flag_total", "artifact_like_fragment_count"
  )
  if (!is.list(artifact_diagnostics) ||
      !identical(
        unname(unlist(artifact_diagnostics$outcome_candidates)),
        expected_artifact_candidates
      ) ||
      !identical(
        as.integer(artifact_diagnostics$row_count),
        as.integer(nrow(features))
      )) {
    abort("Held-out evaluator feature join lacks artifact-outcome provenance")
  }
  artifact_status <- artifact_diagnostics$status
  selected_artifact_column <- artifact_diagnostics$selected_source_column
  derived_artifact_columns <- unname(
    unlist(artifact_diagnostics$derived_columns %||% list())
  )
  if (identical(artifact_status, "source_feature_column_preserved")) {
    if (!(selected_artifact_column %in% expected_artifact_candidates) ||
        length(derived_artifact_columns) != 0L) {
      abort("Preserved artifact-outcome provenance is inconsistent")
    }
  } else if (identical(
    artifact_status, "derived_from_hash_bound_text_rows"
  )) {
    algorithm_path <- artifact_diagnostics$algorithm_source_path
    if (!identical(selected_artifact_column, "surface_flag_total") ||
        !identical(
          derived_artifact_columns,
          c("surface_flag_total", "artifact_like_fragment_count")
        ) ||
        !identical(
          artifact_diagnostics$algorithm_module,
          "rankcloak.revision_statistics"
        ) ||
        !identical(
          artifact_diagnostics$algorithm_function,
          "automated_text_quality_metrics"
        ) ||
        is.null(algorithm_path) ||
        !file.exists(algorithm_path) ||
        !identical(
          sha256_file(algorithm_path),
          artifact_diagnostics$algorithm_source_sha256
        )) {
      abort("Derived artifact-outcome algorithm provenance is inconsistent")
    }
  } else {
    abort("Held-out evaluator feature join has an unknown artifact-outcome status")
  }
  required_artifact_columns <- unique(c(
    selected_artifact_column, derived_artifact_columns
  ))
  if (any(!(required_artifact_columns %in% names(features)))) {
    abort("Joined features lack a manifest-declared artifact outcome")
  }
  for (column in required_artifact_columns) {
    values <- suppressWarnings(as.numeric(features[[column]]))
    if (any(!is.finite(values)) ||
        any(values < 0) ||
        any(abs(values - round(values)) > 1e-12)) {
      abort(sprintf(
        "Joined artifact outcome %s must contain nonnegative integers", column
      ))
    }
  }
  required <- c(
    "source_type", "text_view", "evidence_status", "study_phase",
    "protocol_contract_revision", "result_schema_revision", "transformation_id",
    "heldout_evaluator_log_probability", "heldout_evaluator_model_id",
    "heldout_evaluator_source_record_sha256", "heldout_evaluator_score_scope"
  )
  require_columns(features, required, "joined features")
  if (any(features$source_type != "rankcloak") ||
      any(features$text_view != "full_message") ||
      any(features$evidence_status != PRIMARY_EVIDENCE_STATUS) ||
      any(features$study_phase != PRIMARY_STUDY_PHASE) ||
      any(features$protocol_contract_revision != PROTOCOL_CONTRACT_REVISION) ||
      any(features$result_schema_revision != RESULT_SCHEMA_REVISION) ||
      any(features$transformation_id != "unmodified") ||
      any(features$heldout_evaluator_score_scope != "source_full_message_replicated_across_nested_segment_rows_v1")) {
    abort("Joined features contain rows outside the primary full-message scope")
  }
  scores <- suppressWarnings(as.numeric(features$heldout_evaluator_log_probability))
  if (any(!is.finite(scores))) abort("Joined held-out evaluator scores must be finite")
  split_rows <- split(seq_len(nrow(features)), features$trial_id, drop = TRUE)
  for (indices in split_rows) {
    if (length(unique(scores[indices])) != 1L ||
        length(unique(features$heldout_evaluator_model_id[indices])) != 1L ||
        length(unique(features$heldout_evaluator_source_record_sha256[indices])) != 1L) {
      abort("Nested feature rows disagree on their trial-level evaluator identity")
    }
  }
  invisible(manifest)
}

require_columns <- function(data, columns, label) {
  missing <- setdiff(columns, names(data))
  if (length(missing)) {
    abort(sprintf("%s is missing columns: %s", label, paste(missing, collapse = ", ")))
  }
}

sha256_file <- function(path) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) abort("sha256sum is required for immutable manifests")
  output <- system2(executable, path, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status") %||% 0L
  if (status != 0L || length(output) != 1L) {
    abort(sprintf("Could not hash %s", path))
  }
  digest <- strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
  if (!grepl("^[0-9a-f]{64}$", digest)) abort(sprintf("Invalid SHA-256 for %s", path))
  digest
}

current_script_path <- function() {
  arguments <- commandArgs(trailingOnly = FALSE)
  matches <- grep("^--file=", arguments, value = TRUE)
  if (length(matches) != 1L) abort("Could not resolve the mixed-model driver source path")
  normalizePath(sub("^--file=", "", matches[[1L]]), mustWork = TRUE)
}

normalize_binary <- function(values, label) {
  normalized <- tolower(trimws(as.character(values)))
  result <- rep(NA_integer_, length(normalized))
  result[normalized %in% c("1", "true", "yes")] <- 1L
  result[normalized %in% c("0", "false", "no")] <- 0L
  if (anyNA(result)) {
    examples <- unique(as.character(values[is.na(result)]))
    abort(sprintf("%s is not binary; examples: %s", label, paste(head(examples, 3L), collapse = ", ")))
  }
  result
}

validate_plan <- function(plan) {
  if (!identical(plan$schema_version, "1.0")) abort("Unsupported model-plan schema")
  if (!isTRUE(plan$frozen_before_confirmatory_results)) abort("Plan is not frozen prospectively")
  if (!identical(plan$experimental_unit, "payload_trial")) abort("Experimental unit must be payload_trial")
  if (!isTRUE(plan$segments_as_independent_observations_forbidden)) {
    abort("Plan must forbid treating segments as independent")
  }
  primary_filter <- plan$filters$primary_trials %||% list()
  if (!identical(primary_filter$evidence_status, PRIMARY_EVIDENCE_STATUS) ||
      !identical(primary_filter$study_phase, PRIMARY_STUDY_PHASE) ||
      !identical(primary_filter$protocol_contract_revision, PROTOCOL_CONTRACT_REVISION) ||
      !identical(primary_filter$result_schema_revision, RESULT_SCHEMA_REVISION)) {
    abort("Plan primary filter is not frozen to primary_v2 payload-aware evidence")
  }
  models <- plan$models %||% list()
  if (!length(models)) abort("Plan contains no model specifications")
  for (model in models) {
    if (!identical(model$fixed_effects_fallback, FALSE)) {
      abort(sprintf("Model %s does not explicitly forbid fixed-effects fallback", model$model_id))
    }
    random <- unlist(model$random_intercepts %||% list(), use.names = FALSE)
    if (!all(c("payload_name", "prompt_id") %in% random)) {
      abort(sprintf("Model %s lacks payload/prompt random intercepts", model$model_id))
    }
    formula_text <- model$formula %||% ""
    if (!grepl("model_id \\* protocol_variant", formula_text) ||
        !grepl("model_id \\* prompt_category", formula_text) ||
        !grepl("protocol_variant \\* prompt_category", formula_text)) {
      abort(sprintf("Model %s lacks prespecified model/prompt/codec interactions", model$model_id))
    }
  }
  families <- plan$contrast_families %||% list()
  if (!length(families)) abort("Plan contains no contrast families")
  for (family in families) {
    if (!startsWith(tolower(family$adjustment %||% ""), "holm")) {
      abort(sprintf("Contrast family %s is not Holm controlled", family$family_id))
    }
  }
  invisible(TRUE)
}

verify_locked_environment <- function(lock, lock_path) {
  expected_r <- lock$r$version
  found_r <- paste(R.version$major, R.version$minor, sep = ".")
  if (!identical(found_r, expected_r)) {
    abort(sprintf("R version mismatch: expected %s, found %s", expected_r, R.version.string))
  }
  repo_root <- dirname(dirname(dirname(normalizePath(lock_path, mustWork = TRUE))))
  resolution <- lock$library$resolution_order %||% list()
  if (!length(resolution)) abort("R lock has no library resolution order")
  libraries <- list()
  for (entry in resolution) {
    kind <- entry$path_kind
    if (identical(kind, "r_runtime_default")) next
    path <- if (identical(kind, "repository_relative")) {
      file.path(repo_root, entry$path)
    } else if (identical(kind, "absolute_existing")) {
      entry$path
    } else {
      abort(sprintf("Unknown R library path kind: %s", kind))
    }
    libraries[[entry$role]] <- normalizePath(path, mustWork = TRUE)
  }
  .libPaths(unique(c(unlist(libraries, use.names = FALSE), .libPaths())))
  expected_packages <- lock$packages
  package_rows <- list()
  for (package_name in names(expected_packages)) {
    expected_version <- expected_packages[[package_name]]$version
    expected_role <- expected_packages[[package_name]]$required_library_role
    expected_library <- libraries[[expected_role]]
    if (is.null(expected_library)) {
      abort(sprintf("Package %s names undeclared library role %s", package_name, expected_role))
    }
    location <- find.package(package_name, quiet = TRUE)
    if (!nzchar(location)) {
      abort(sprintf(
        "Locked package missing: %s %s",
        package_name, expected_version
      ))
    }
    found_version <- as.character(utils::packageVersion(package_name))
    if (utils::compareVersion(found_version, expected_version) != 0L) {
      abort(sprintf(
        "Locked package mismatch for %s: expected %s, found %s",
        package_name, expected_version, found_version
      ))
    }
    resolved <- normalizePath(location, mustWork = TRUE)
    prefix <- paste0(expected_library, .Platform$file.sep)
    if (!startsWith(resolved, prefix)) {
      abort(sprintf(
        "Package %s resolved outside declared role %s: %s",
        package_name, expected_role, resolved
      ))
    }
    package_rows[[package_name]] <- list(
      version = found_version,
      path = resolved,
      required_library_role = expected_role,
      expected_version = expected_version
    )
  }
  list(
    r_version = found_r,
    r_version_string = R.version.string,
    ordered_library_roles = libraries,
    resolved_lib_paths = as.list(.libPaths()),
    packages = package_rows,
    fixed_effects_fallback = FALSE
  )
}

filter_primary_trials <- function(trials, plan) {
  required <- c(
    "trial_id", "record_type", "evidence_status", "study_phase", "replay_mode",
    "protocol_contract_revision", "result_schema_revision",
    "exact_rank_replay", "exact_payload_recovery",
    "recovery_outcome_semantics", "exact_recovery", "model_id",
    "protocol_variant", "prompt_id", "prompt_category", "payload_name",
    "payload_class"
  )
  require_columns(trials, required, "trials")
  filter <- plan$filters$primary_trials
  selected <- trials[
    trials$evidence_status == filter$evidence_status &
      trials$study_phase == filter$study_phase &
      trials$protocol_contract_revision == filter$protocol_contract_revision &
      trials$result_schema_revision == filter$result_schema_revision &
      trials$record_type == filter$record_type &
      trials$replay_mode == filter$replay_mode,
    ,
    drop = FALSE
  ]
  if (nrow(selected) == 0L) abort("No rows satisfy the frozen primary-trial filter")
  if (anyDuplicated(selected$trial_id)) {
    abort("Primary recovery has duplicate trial IDs; segments/replays cannot be independent rows")
  }
  selected$exact_rank_replay <- normalize_binary(
    selected$exact_rank_replay, "exact_rank_replay"
  )
  selected$exact_payload_recovery <- normalize_binary(
    selected$exact_payload_recovery, "exact_payload_recovery"
  )
  selected$exact_recovery <- normalize_binary(
    selected$exact_recovery, "exact_recovery"
  )
  if (any(selected$exact_recovery != selected$exact_payload_recovery)) {
    abort("exact_recovery compatibility alias differs from exact_payload_recovery")
  }
  if (any(is.na(selected$recovery_outcome_semantics)) ||
      any(selected$recovery_outcome_semantics != PAYLOAD_RECOVERY_SEMANTICS)) {
    abort("Primary recovery rows have ambiguous recovery_outcome_semantics")
  }
  direct_rows <- sum(selected$protocol_variant == "direct_subword_calgacus")
  attr(selected, "payload_fidelity_contract") <- list(
    contract_version = PROTOCOL_CONTRACT_REVISION,
    result_schema_revision = RESULT_SCHEMA_REVISION,
    semantics = PAYLOAD_RECOVERY_SEMANTICS,
    primary_outcome = "exact_payload_recovery",
    compatibility_alias = "exact_recovery",
    alias_equality_validated = TRUE,
    exact_rank_replay_role = "diagnostic_only",
    direct_rows = direct_rows,
    direct_rows_contract_verified = direct_rows
  )
  selected
}

apply_reference_levels <- function(data, plan) {
  references <- plan$reference_levels
  for (column in names(references)) {
    if (!column %in% names(data)) next
    reference <- references[[column]]
    values <- unique(as.character(data[[column]][!is.na(data[[column]])]))
    if (!reference %in% values) {
      abort(sprintf("Prespecified reference %s=%s is absent", column, reference))
    }
    data[[column]] <- stats::relevel(factor(data[[column]]), ref = reference)
  }
  data
}

metadata_columns <- c(
  "trial_id", "model_id", "protocol_variant", "prompt_id",
  "prompt_category", "payload_name", "payload_class"
)

trial_metadata <- function(primary) {
  result <- primary[, metadata_columns, drop = FALSE]
  if (anyDuplicated(result$trial_id)) abort("Trial metadata is not unique")
  result
}

collapse_feature_rows <- function(features, primary, plan) {
  if (is.null(features)) return(NULL)
  required <- c("trial_id", "segment_index", "text_view", "token_count")
  require_columns(features, required, "features")
  selected <- features[
    features$trial_id %in% primary$trial_id &
      features$text_view == plan$filters$primary_features$text_view,
    ,
    drop = FALSE
  ]
  if (nrow(selected) == 0L) abort("No full-message primary feature rows remain")
  identity <- paste(selected$trial_id, selected$text_view, selected$segment_index, sep = "\r")
  if (anyDuplicated(identity)) abort("Duplicate segment rows in features")
  candidates <- unlist(
    Filter(
      function(model) identical(model$model_id, "primary_artifact_counts"),
      plan$models
    )[[1L]]$outcome_candidates,
    use.names = FALSE
  )
  artifact_column <- candidates[candidates %in% names(selected)][1L]
  split_rows <- split(seq_len(nrow(selected)), selected$trial_id, drop = TRUE)
  output <- lapply(names(split_rows), function(trial_id) {
    rows <- selected[split_rows[[trial_id]], , drop = FALSE]
    tokens <- suppressWarnings(as.numeric(rows$token_count))
    if (any(!is.finite(tokens)) || any(tokens < 0)) abort("Feature token_count must be finite and nonnegative")
    result <- data.frame(
      trial_id = trial_id,
      token_count = sum(tokens),
      nested_segment_count = nrow(rows),
      stringsAsFactors = FALSE
    )
    if (!is.na(artifact_column)) {
      counts <- suppressWarnings(as.numeric(rows[[artifact_column]]))
      if (any(!is.finite(counts)) || any(counts < 0) || any(abs(counts - round(counts)) > 1e-8)) {
        abort(sprintf("%s must contain nonnegative integer counts", artifact_column))
      }
      result$artifact_count <- sum(round(counts))
      result$artifact_count_source_column <- artifact_column
    }
    weighted_mean <- function(column) {
      values <- suppressWarnings(as.numeric(rows[[column]]))
      valid <- is.finite(values) & is.finite(tokens) & tokens > 0
      if (!any(valid)) return(NA_real_)
      stats::weighted.mean(values[valid], tokens[valid])
    }
    if ("mean_log_probability" %in% names(rows)) {
      result$mean_log_probability <- weighted_mean("mean_log_probability")
    }
    if ("heldout_evaluator_log_probability" %in% names(rows)) {
      result$heldout_evaluator_log_probability <- weighted_mean(
        "heldout_evaluator_log_probability"
      )
    }
    result
  })
  collapsed <- do.call(rbind, output)
  rownames(collapsed) <- NULL
  merged <- merge(collapsed, trial_metadata(primary), by = "trial_id", all.x = TRUE, sort = FALSE)
  if (nrow(merged) != length(split_rows) || anyNA(merged$payload_name)) {
    abort("Feature-to-trial metadata join failed")
  }
  merged
}

prepare_runtime <- function(runtime, primary, plan) {
  if (is.null(runtime)) return(NULL)
  require_columns(runtime, c("trial_id", "runtime_scope"), "runtime")
  selected <- runtime[
    runtime$runtime_scope == plan$filters$runtime$runtime_scope &
      runtime$trial_id %in% primary$trial_id,
    ,
    drop = FALSE
  ]
  if (nrow(selected) == 0L) abort("No primary trial runtime rows remain")
  if (anyDuplicated(selected$trial_id)) abort("Runtime contains duplicate primary trial IDs")
  keep <- setdiff(names(selected), setdiff(metadata_columns, "trial_id"))
  merged <- merge(
    selected[, keep, drop = FALSE],
    trial_metadata(primary),
    by = "trial_id",
    all.x = TRUE,
    sort = FALSE
  )
  if (nrow(merged) != nrow(selected) || anyNA(merged$payload_name)) {
    abort("Runtime-to-trial metadata join failed")
  }
  merged
}

complete_model_data <- function(data, formula_value, label) {
  variables <- all.vars(formula_value)
  require_columns(data, variables, label)
  complete <- stats::complete.cases(data[, variables, drop = FALSE])
  removed <- sum(!complete)
  result <- data[complete, , drop = FALSE]
  if (nrow(result) == 0L) abort(sprintf("%s has no complete model rows", label))
  attr(result, "excluded_incomplete_rows") <- removed
  result
}

wilson_interval <- function(successes, total, confidence = 0.95) {
  if (total <= 0L || successes < 0L || successes > total) abort("Invalid Wilson counts")
  z <- stats::qnorm(0.5 + confidence / 2)
  proportion <- successes / total
  denominator <- 1 + z^2 / total
  centre <- proportion + z^2 / (2 * total)
  radius <- z * sqrt(proportion * (1 - proportion) / total + z^2 / (4 * total^2))
  c(
    low = max(0, (centre - radius) / denominator),
    high = min(1, (centre + radius) / denominator)
  )
}

wilson_rows <- function(data, confidence) {
  group_sets <- list(
    character(),
    "model_id",
    "protocol_variant",
    "prompt_category",
    "payload_class"
  )
  rows <- list()
  for (columns in group_sets) {
    groups <- if (!length(columns)) {
      list(overall = seq_len(nrow(data)))
    } else {
      split(seq_len(nrow(data)), interaction(data[, columns, drop = FALSE], drop = TRUE, lex.order = TRUE))
    }
    for (indices in groups) {
      cell <- data[indices, , drop = FALSE]
      successes <- sum(cell$exact_recovery)
      interval <- wilson_interval(successes, nrow(cell), confidence)
      row <- data.frame(
        grouping = if (length(columns)) paste(columns, collapse = "+") else "overall",
        group_value = if (length(columns)) paste(as.character(cell[1L, columns, drop = TRUE]), collapse = "|") else "overall",
        recovery_outcome = "exact_payload_recovery",
        recovery_outcome_semantics = PAYLOAD_RECOVERY_SEMANTICS,
        exact_recovery_compatibility_alias = TRUE,
        exact_rank_replay_diagnostic_only = TRUE,
        successes = successes,
        total = nrow(cell),
        estimate = successes / nrow(cell),
        confidence_level = confidence,
        wilson_ci_low = interval[["low"]],
        wilson_ci_high = interval[["high"]],
        stringsAsFactors = FALSE
      )
      rows[[length(rows) + 1L]] <- row
    }
  }
  do.call(rbind, rows)
}

capture_fit <- function(expression) {
  warnings <- character()
  fit <- withCallingHandlers(
    expression,
    warning = function(condition) {
      warnings <<- c(warnings, conditionMessage(condition))
      invokeRestart("muffleWarning")
    }
  )
  list(fit = fit, warnings = unique(warnings))
}

fit_diagnostics <- function(fit, model_id, warnings, plan, extra = list()) {
  optinfo <- fit@optinfo
  messages <- optinfo$conv$lme4$messages %||% character()
  gradient <- optinfo$derivs$gradient %||% numeric()
  hessian <- optinfo$derivs$Hessian %||% matrix(numeric(), 0L, 0L)
  eigenvalues <- if (length(hessian)) {
    tryCatch(eigen(hessian, symmetric = TRUE, only.values = TRUE)$values, error = function(e) numeric())
  } else numeric()
  fixed <- lme4::fixef(fit)
  thresholds <- plan$diagnostics
  result <- list(
    model_id = model_id,
    converged = length(messages) == 0L,
    convergence_messages = as.list(as.character(messages)),
    warnings = as.list(warnings),
    singular = lme4::isSingular(fit, tol = thresholds$singular_tolerance),
    singular_tolerance = thresholds$singular_tolerance,
    maximum_absolute_gradient = if (length(gradient)) max(abs(gradient)) else NULL,
    gradient_threshold = thresholds$maximum_absolute_gradient,
    minimum_hessian_eigenvalue = if (length(eigenvalues)) min(eigenvalues) else NULL,
    hessian_threshold = thresholds$minimum_hessian_eigenvalue,
    maximum_absolute_fixed_coefficient = if (length(fixed)) max(abs(fixed)) else NULL,
    potential_separation = any(abs(fixed) >= thresholds$potential_separation_absolute_coefficient),
    separation_coefficient_threshold = thresholds$potential_separation_absolute_coefficient,
    rank_deficient_columns = as.list(names(attr(lme4::getME(fit, "X"), "col.dropped") %||% integer())),
    fixed_effects_fallback = FALSE
  )
  modifyList(result, extra)
}

tidy_fit <- function(fit, model_id, family_name, formula_text) {
  matrix <- as.data.frame(summary(fit)$coefficients)
  terms <- rownames(matrix)
  estimate <- matrix[[1L]]
  standard_error <- matrix[[2L]]
  statistic <- if (ncol(matrix) >= 3L) matrix[[3L]] else rep(NA_real_, nrow(matrix))
  p_column <- grep("^Pr\\(", names(matrix), value = TRUE)
  p_value <- if (length(p_column)) matrix[[p_column[[1L]]]] else rep(NA_real_, nrow(matrix))
  data.frame(
    model_id = model_id,
    backend = "R_lme4",
    family = family_name,
    formula = formula_text,
    term = terms,
    estimate = estimate,
    standard_error = standard_error,
    statistic = statistic,
    p_value_raw = p_value,
    ci_low = estimate - stats::qnorm(0.975) * standard_error,
    ci_high = estimate + stats::qnorm(0.975) * standard_error,
    fixed_effects_fallback = FALSE,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

matching_contrast_families <- function(plan, model_id) {
  Filter(function(family) {
    exact <- family$model_id
    pattern <- family$model_id_pattern
    (!is.null(exact) && identical(exact, model_id)) ||
      (!is.null(pattern) && grepl(pattern, model_id))
  }, plan$contrast_families %||% list())
}

fit_contrasts <- function(fit, model_id, plan, default_scale) {
  families <- matching_contrast_families(plan, model_id)
  if (!length(families)) return(NULL)
  rows <- list()
  for (family in families) {
    factor_name <- family$factor
    by <- unlist(family$by %||% list(), use.names = FALSE)
    right <- factor_name
    if (length(by)) right <- paste(right, "|", paste(by, collapse = "*"))
    specification <- stats::as.formula(paste("~", right))
    scale <- family$scale %||% default_scale
    emmeans_object <- emmeans::emmeans(
      fit,
      specs = specification,
      type = if (identical(scale, "response")) "response" else "link"
    )
    pairs <- as.data.frame(
      summary(
        emmeans::contrast(emmeans_object, method = "pairwise", adjust = "none"),
        infer = c(TRUE, TRUE),
        type = if (identical(scale, "response")) "response" else "link"
      )
    )
    if (!nrow(pairs)) next
    p_column <- if ("p.value" %in% names(pairs)) "p.value" else NULL
    estimate_column <- intersect(c("estimate", "odds.ratio", "ratio", "response"), names(pairs))[1L]
    if (is.na(estimate_column)) estimate_column <- names(pairs)[1L]
    raw <- if (!is.null(p_column)) pairs[[p_column]] else rep(NA_real_, nrow(pairs))
    adjusted <- rep(NA_real_, length(raw))
    finite <- is.finite(raw)
    adjusted[finite] <- stats::p.adjust(raw[finite], method = "holm")
    row <- data.frame(
      model_id = model_id,
      stratum_model_id = NA_character_,
      multiplicity_family = family$family_id,
      contrast = as.character(pairs$contrast),
      estimate = suppressWarnings(as.numeric(pairs[[estimate_column]])),
      standard_error = if ("SE" %in% names(pairs)) pairs$SE else NA_real_,
      statistic = if ("z.ratio" %in% names(pairs)) pairs$z.ratio else if ("t.ratio" %in% names(pairs)) pairs$t.ratio else NA_real_,
      p_value_raw = raw,
      p_value_holm = adjusted,
      ci_low = if ("lower.CL" %in% names(pairs)) pairs$lower.CL else if ("asymp.LCL" %in% names(pairs)) pairs$asymp.LCL else NA_real_,
      ci_high = if ("upper.CL" %in% names(pairs)) pairs$upper.CL else if ("asymp.UCL" %in% names(pairs)) pairs$asymp.UCL else NA_real_,
      adjustment = "holm",
      scale = scale,
      fixed_effects_fallback = FALSE,
      stringsAsFactors = FALSE
    )
    for (column in by) {
      if (column %in% names(pairs)) {
        stratum_column <- if (identical(column, "model_id")) {
          "stratum_model_id"
        } else {
          paste0("stratum_", column)
        }
        row[[stratum_column]] <- pairs[[column]]
      }
    }
    rows[[length(rows) + 1L]] <- row
  }
  if (!length(rows)) NULL else do.call(rbind, rows)
}

model_spec <- function(plan, model_id) {
  matches <- Filter(function(model) identical(model$model_id, model_id), plan$models)
  if (length(matches) != 1L) abort(sprintf("Expected one model spec for %s", model_id))
  matches[[1L]]
}

glmer_control <- function(plan) {
  lme4::glmerControl(
    optimizer = plan$diagnostics$optimizer,
    optCtrl = list(maxfun = plan$diagnostics$maxfun),
    calc.derivs = TRUE
  )
}

lmer_control <- function(plan) {
  lme4::lmerControl(
    optimizer = plan$diagnostics$optimizer,
    optCtrl = list(maxfun = plan$diagnostics$maxfun),
    calc.derivs = TRUE
  )
}

fit_recovery_model <- function(primary, plan) {
  spec <- model_spec(plan, "primary_exact_recovery")
  formula_value <- stats::as.formula(spec$formula)
  data <- complete_model_data(apply_reference_levels(primary, plan), formula_value, spec$model_id)
  wilson <- wilson_rows(data, plan$confidence_level)
  unique_outcome <- unique(data$exact_recovery)
  if (length(unique_outcome) == 1L) {
    direction <- if (unique_outcome[[1L]] == 1L) "all_success" else "all_failure"
    status <- list(
      model_id = spec$model_id,
      status = paste0("not_fitted_complete_outcome_separation_", direction),
      reason = "The binomial likelihood has no outcome variation; fixed effects and random-effect variances are unidentified.",
      all_success_or_zero_policy = spec$all_success_or_zero_policy,
      coefficient_rows = 0L,
      recovery_outcome = "exact_payload_recovery",
      recovery_outcome_semantics = PAYLOAD_RECOVERY_SEMANTICS,
      fixed_effects_fallback = FALSE
    )
    diagnostic <- list(
      model_id = spec$model_id,
      complete_outcome_separation = TRUE,
      outcome_pattern = direction,
      glmer_attempted = FALSE,
      wilson_sensitivity_reported = TRUE,
      recovery_outcome = "exact_payload_recovery",
      recovery_outcome_semantics = PAYLOAD_RECOVERY_SEMANTICS,
      exact_rank_replay_role = "diagnostic_only",
      fixed_effects_fallback = FALSE
    )
    return(list(fit = NULL, coefficients = NULL, contrasts = NULL, diagnostics = diagnostic, status = status, wilson = wilson))
  }
  captured <- capture_fit(
    lme4::glmer(
      formula_value,
      data = data,
      family = stats::binomial(link = "logit"),
      control = glmer_control(plan),
      nAGQ = plan$diagnostics$nAGQ,
      na.action = stats::na.fail
    )
  )
  diagnostic <- fit_diagnostics(captured$fit, spec$model_id, captured$warnings, plan)
  status <- list(
    model_id = spec$model_id,
    status = if (isTRUE(diagnostic$converged) && !isTRUE(diagnostic$potential_separation)) "completed" else "completed_with_diagnostic_warning",
    coefficient_rows = length(lme4::fixef(captured$fit)),
    excluded_incomplete_rows = attr(data, "excluded_incomplete_rows"),
    fixed_effects_fallback = FALSE
  )
  coefficients <- tidy_fit(
    captured$fit, spec$model_id, "binomial_logit", spec$formula
  )
  coefficients$recovery_outcome <- "exact_payload_recovery"
  coefficients$recovery_outcome_semantics <- PAYLOAD_RECOVERY_SEMANTICS
  contrasts <- fit_contrasts(captured$fit, spec$model_id, plan, "response")
  if (!is.null(contrasts)) {
    contrasts$recovery_outcome <- "exact_payload_recovery"
    contrasts$recovery_outcome_semantics <- PAYLOAD_RECOVERY_SEMANTICS
  }
  diagnostic$recovery_outcome <- "exact_payload_recovery"
  diagnostic$recovery_outcome_semantics <- PAYLOAD_RECOVERY_SEMANTICS
  diagnostic$exact_rank_replay_role <- "diagnostic_only"
  status$recovery_outcome <- "exact_payload_recovery"
  status$recovery_outcome_semantics <- PAYLOAD_RECOVERY_SEMANTICS
  list(
    fit = captured$fit,
    coefficients = coefficients,
    contrasts = contrasts,
    diagnostics = diagnostic,
    status = status,
    wilson = wilson
  )
}

poisson_dispersion <- function(fit) {
  residuals <- stats::residuals(fit, type = "pearson")
  rdf <- stats::nobs(fit) - length(lme4::fixef(fit))
  if (rdf <= 0L) return(NA_real_)
  sum(residuals^2) / rdf
}

fit_artifact_model <- function(features, plan) {
  spec <- model_spec(plan, "primary_artifact_counts")
  if (is.null(features) || !"artifact_count" %in% names(features)) {
    abort("Artifact-count model requires one prespecified artifact count column in features")
  }
  if (all(features$artifact_count == 0L)) {
    status <- list(
      model_id = spec$model_id,
      status = "not_fitted_all_zero_counts",
      reason = "Negative-binomial mean/dispersion effects are unidentified when every count is zero.",
      fixed_effects_fallback = FALSE
    )
    return(list(
      fit = NULL,
      coefficients = NULL,
      contrasts = NULL,
      diagnostics = list(model_id = spec$model_id, all_zero_counts = TRUE, glmer_nb_attempted = FALSE, fixed_effects_fallback = FALSE),
      dispersion = data.frame(
        model_id = spec$model_id,
        poisson_dispersion_ratio = NA_real_,
        threshold = plan$diagnostics$poisson_overdispersion_threshold,
        classification = "all_zero_counts_not_evaluable",
        negative_binomial_remains_primary = TRUE,
        stringsAsFactors = FALSE
      ),
      status = status
    ))
  }
  formula_value <- stats::as.formula(spec$formula)
  data <- complete_model_data(apply_reference_levels(features, plan), formula_value, spec$model_id)
  if (any(data$token_count <= 0)) abort("Artifact-count offset requires positive token_count")
  poisson_captured <- capture_fit(
    lme4::glmer(
      formula_value,
      data = data,
      family = stats::poisson(link = "log"),
      control = glmer_control(plan),
      nAGQ = plan$diagnostics$nAGQ,
      na.action = stats::na.fail
    )
  )
  dispersion_value <- poisson_dispersion(poisson_captured$fit)
  classification <- if (!is.finite(dispersion_value)) {
    "not_evaluable"
  } else if (dispersion_value > plan$diagnostics$poisson_overdispersion_threshold) {
    "overdispersed"
  } else if (dispersion_value < plan$diagnostics$poisson_underdispersion_threshold) {
    "underdispersed"
  } else {
    "approximately_equidispersed"
  }
  nb_captured <- capture_fit(
    lme4::glmer.nb(
      formula_value,
      data = data,
      control = glmer_control(plan),
      nAGQ = plan$diagnostics$nAGQ,
      na.action = stats::na.fail
    )
  )
  theta <- tryCatch(lme4::getME(nb_captured$fit, "glmer.nb.theta"), error = function(e) NA_real_)
  diagnostic <- fit_diagnostics(
    nb_captured$fit,
    spec$model_id,
    c(poisson_captured$warnings, nb_captured$warnings),
    plan,
    extra = list(
      negative_binomial_theta = theta,
      poisson_dispersion_ratio = dispersion_value,
      poisson_dispersion_classification = classification,
      poisson_check_is_diagnostic_not_selection = TRUE
    )
  )
  status <- list(
    model_id = spec$model_id,
    status = if (isTRUE(diagnostic$converged)) "completed" else "completed_with_diagnostic_warning",
    coefficient_rows = length(lme4::fixef(nb_captured$fit)),
    excluded_incomplete_rows = attr(data, "excluded_incomplete_rows"),
    fixed_effects_fallback = FALSE
  )
  list(
    fit = nb_captured$fit,
    coefficients = tidy_fit(nb_captured$fit, spec$model_id, "negative_binomial_log", spec$formula),
    contrasts = fit_contrasts(nb_captured$fit, spec$model_id, plan, "response"),
    diagnostics = diagnostic,
    dispersion = data.frame(
      model_id = spec$model_id,
      poisson_dispersion_ratio = dispersion_value,
      underdispersion_threshold = plan$diagnostics$poisson_underdispersion_threshold,
      overdispersion_threshold = plan$diagnostics$poisson_overdispersion_threshold,
      classification = classification,
      negative_binomial_remains_primary = TRUE,
      stringsAsFactors = FALSE
    ),
    status = status
  )
}

fit_continuous_model <- function(data, spec, plan) {
  if (is.null(data) || !spec$outcome %in% names(data)) {
    if (isTRUE(spec$required_when_column_available)) {
      return(list(
        fit = NULL,
        coefficients = NULL,
        contrasts = NULL,
        diagnostics = list(model_id = spec$model_id, column_available = FALSE, fixed_effects_fallback = FALSE),
        status = list(
          model_id = spec$model_id,
          status = "not_run_column_not_available",
          required_when_column_available = TRUE,
          fixed_effects_fallback = FALSE
        )
      ))
    }
    abort(sprintf("Required continuous outcome is unavailable: %s", spec$outcome))
  }
  formula_value <- stats::as.formula(spec$formula)
  model_data <- complete_model_data(apply_reference_levels(data, plan), formula_value, spec$model_id)
  if (grepl("^log\\(", spec$formula) && any(model_data[[spec$outcome]] <= 0)) {
    abort(sprintf("%s requires positive values for log transform", spec$model_id))
  }
  captured <- capture_fit(
    lme4::lmer(
      formula_value,
      data = model_data,
      REML = FALSE,
      control = lmer_control(plan),
      na.action = stats::na.fail
    )
  )
  diagnostic <- fit_diagnostics(captured$fit, spec$model_id, captured$warnings, plan)
  status <- list(
    model_id = spec$model_id,
    status = if (isTRUE(diagnostic$converged)) "completed" else "completed_with_diagnostic_warning",
    coefficient_rows = length(lme4::fixef(captured$fit)),
    excluded_incomplete_rows = attr(model_data, "excluded_incomplete_rows"),
    fixed_effects_fallback = FALSE
  )
  list(
    fit = captured$fit,
    coefficients = tidy_fit(captured$fit, spec$model_id, spec$family, spec$formula),
    contrasts = fit_contrasts(captured$fit, spec$model_id, plan, "model"),
    diagnostics = diagnostic,
    status = status
  )
}

rbind_optional <- function(values, columns) {
  present <- Filter(function(value) !is.null(value) && nrow(value) > 0L, values)
  if (!length(present)) {
    result <- as.data.frame(setNames(replicate(length(columns), logical(0L), simplify = FALSE), columns))
    return(result)
  }
  all_columns <- unique(unlist(lapply(present, names), use.names = FALSE))
  normalized <- lapply(present, function(value) {
    for (column in setdiff(all_columns, names(value))) value[[column]] <- NA
    value[, all_columns, drop = FALSE]
  })
  do.call(rbind, normalized)
}

write_csv <- function(data, path) {
  utils::write.csv(data, path, row.names = FALSE, na = "")
}

output_file_entry <- function(path, stage_directory) {
  data <- if (grepl("\\.csv$", path)) utils::read.csv(path, check.names = FALSE) else NULL
  list(
    path = basename(path),
    size_bytes = unname(file.info(path)$size),
    sha256 = sha256_file(path),
    row_count = if (is.null(data)) NULL else nrow(data)
  )
}

stage_outputs <- function(output_directory, writer) {
  output_directory <- normalizePath(output_directory, mustWork = FALSE)
  if (file.exists(output_directory)) abort(sprintf("Output already exists: %s", output_directory))
  parent <- dirname(output_directory)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  stage <- tempfile(pattern = paste0(".", basename(output_directory), ".staging-"), tmpdir = parent)
  if (!dir.create(stage, recursive = FALSE)) abort("Could not create staging directory")
  committed <- FALSE
  on.exit(if (!committed && dir.exists(stage)) unlink(stage, recursive = TRUE, force = TRUE), add = TRUE)
  writer(stage)
  if (!file.rename(stage, output_directory)) abort("Atomic output-directory commit failed")
  committed <- TRUE
  invisible(output_directory)
}

validation_statuses <- function(plan) {
  statuses <- lapply(plan$models, function(spec) {
    list(
      model_id = spec$model_id,
      status = "not_fitted_validation_only",
      fixed_effects_fallback = FALSE
    )
  })
  statuses[[length(statuses) + 1L]] <- list(
    model_id = plan$human_model$model_id,
    status = plan$human_model$status,
    recruitment_authorized = plan$human_model$recruitment_authorized,
    fixed_effects_fallback = FALSE
  )
  statuses
}

main <- function(arguments = commandArgs(trailingOnly = TRUE)) {
  options <- parse_cli(arguments)
  plan_path <- require_option(options, "plan")
  trials_path <- require_option(options, "trials")
  output_directory <- require_option(options, "output_dir")
  lock_path <- options$environment_lock %||% file.path(dirname(plan_path), "r_environment.lock.json")
  plan <- read_json(plan_path)
  lock <- read_json(lock_path)
  validate_plan(plan)
  found_r <- paste(R.version$major, R.version$minor, sep = ".")
  if (!identical(found_r, lock$r$version)) {
    abort(sprintf("R version mismatch: expected %s, found %s", lock$r$version, found_r))
  }

  trials <- read_csv_required(trials_path, "trials")
  features <- read_csv_optional(options$features, "features")
  runtime <- read_csv_optional(options$runtime, "runtime")
  detectors <- read_csv_optional(options$detectors, "detectors")
  primary <- filter_primary_trials(trials, plan)
  if (!isTRUE(options$validate_only) && is.null(options$feature_join_manifest)) {
    abort("Confirmatory execution requires --feature-join-manifest")
  }
  if (!is.null(options$feature_join_manifest)) {
    if (is.null(features)) abort("--feature-join-manifest requires --features")
    verify_feature_join_manifest(options$feature_join_manifest, options$features, features)
  }
  payload_fidelity_contract <- attr(primary, "payload_fidelity_contract")
  collapsed_features <- collapse_feature_rows(features, primary, plan)
  joined_runtime <- prepare_runtime(runtime, primary, plan)
  wilson <- wilson_rows(primary, plan$confidence_level)

  input_paths <- Filter(
    Negate(is.null),
    list(
      driver_source = current_script_path(),
      plan = plan_path,
      environment_lock = lock_path,
      trials = trials_path,
      features = options$features,
      feature_join_manifest = options$feature_join_manifest,
      runtime = options$runtime,
      detectors = options$detectors
    )
  )
  input_manifest <- lapply(names(input_paths), function(name) {
    path <- normalizePath(input_paths[[name]], mustWork = TRUE)
    list(role = name, path = path, size_bytes = unname(file.info(path)$size), sha256 = sha256_file(path))
  })

  environment <- if (isTRUE(options$validate_only)) {
    list(
      r_version = found_r,
      locked_package_check = "not_run_validation_only",
      fixed_effects_fallback = FALSE
    )
  } else {
    verify_locked_environment(lock, lock_path)
  }

  coefficients <- list()
  contrasts <- list()
  diagnostics <- list()
  statuses <- list()
  dispersion <- list()

  if (isTRUE(options$validate_only)) {
    statuses <- validation_statuses(plan)
    diagnostics[[1L]] <- list(
      validation_only = TRUE,
      plan_valid = TRUE,
      primary_trial_rows = nrow(primary),
      independent_payloads = length(unique(primary$payload_name)),
      input_feature_rows = if (is.null(features)) 0L else nrow(features),
      collapsed_feature_trial_rows = if (is.null(collapsed_features)) 0L else nrow(collapsed_features),
      segments_as_independent_observations = FALSE,
      runtime_trial_rows = if (is.null(joined_runtime)) 0L else nrow(joined_runtime),
      detector_rows_inspected_only = if (is.null(detectors)) 0L else nrow(detectors),
      fixed_effects_fallback = FALSE
    )
  } else {
    recovery <- fit_recovery_model(primary, plan)
    coefficients[[length(coefficients) + 1L]] <- recovery$coefficients
    contrasts[[length(contrasts) + 1L]] <- recovery$contrasts
    diagnostics[[length(diagnostics) + 1L]] <- recovery$diagnostics
    statuses[[length(statuses) + 1L]] <- recovery$status
    wilson <- recovery$wilson

    artifact <- fit_artifact_model(collapsed_features, plan)
    coefficients[[length(coefficients) + 1L]] <- artifact$coefficients
    contrasts[[length(contrasts) + 1L]] <- artifact$contrasts
    diagnostics[[length(diagnostics) + 1L]] <- artifact$diagnostics
    statuses[[length(statuses) + 1L]] <- artifact$status
    dispersion[[length(dispersion) + 1L]] <- artifact$dispersion

    continuous_ids <- c(
      "primary_effective_artifact_rate",
      "primary_cover_log_probability",
      "primary_heldout_evaluator_log_probability",
      "primary_payload_throughput"
    )
    data_by_id <- list(
      primary_effective_artifact_rate = primary,
      primary_cover_log_probability = collapsed_features,
      primary_heldout_evaluator_log_probability = collapsed_features,
      primary_payload_throughput = joined_runtime
    )
    for (model_id in continuous_ids) {
      result <- fit_continuous_model(data_by_id[[model_id]], model_spec(plan, model_id), plan)
      coefficients[[length(coefficients) + 1L]] <- result$coefficients
      contrasts[[length(contrasts) + 1L]] <- result$contrasts
      diagnostics[[length(diagnostics) + 1L]] <- result$diagnostics
      statuses[[length(statuses) + 1L]] <- result$status
    }
    statuses[[length(statuses) + 1L]] <- list(
      model_id = plan$human_model$model_id,
      status = plan$human_model$status,
      recruitment_authorized = plan$human_model$recruitment_authorized,
      fixed_effects_fallback = FALSE
    )
  }

  coefficient_table <- rbind_optional(
    coefficients,
    c("model_id", "backend", "family", "formula", "term", "estimate", "standard_error", "statistic", "p_value_raw", "ci_low", "ci_high", "fixed_effects_fallback")
  )
  contrast_table <- rbind_optional(
    contrasts,
    c("model_id", "stratum_model_id", "multiplicity_family", "contrast", "estimate", "standard_error", "statistic", "p_value_raw", "p_value_holm", "ci_low", "ci_high", "adjustment", "scale", "fixed_effects_fallback")
  )
  dispersion_table <- rbind_optional(
    dispersion,
    c("model_id", "poisson_dispersion_ratio", "underdispersion_threshold", "overdispersion_threshold", "classification", "negative_binomial_remains_primary")
  )

  stage_outputs(output_directory, function(stage) {
    paths <- list(
      coefficients = file.path(stage, "mixed_model_coefficients.csv"),
      contrasts = file.path(stage, "mixed_model_contrasts.csv"),
      diagnostics = file.path(stage, "mixed_model_diagnostics.json"),
      wilson = file.path(stage, "recovery_wilson_sensitivity.csv"),
      dispersion = file.path(stage, "poisson_dispersion_check.csv"),
      status = file.path(stage, "model_status.json")
    )
    write_csv(coefficient_table, paths$coefficients)
    write_csv(contrast_table, paths$contrasts)
    write_json(diagnostics, paths$diagnostics)
    write_csv(wilson, paths$wilson)
    write_csv(dispersion_table, paths$dispersion)
    write_json(statuses, paths$status)
    files <- lapply(paths, output_file_entry, stage_directory = stage)
    manifest <- list(
      schema_version = "1.0",
      manifest_type = "rankcloak_revision_v1_mixed_model_run",
      plan_id = plan$plan_id,
      plan_sha256 = sha256_file(plan_path),
      environment_lock_sha256 = sha256_file(lock_path),
      validation_only = isTRUE(options$validate_only),
      analysis_unit = "payload_trial",
      segments_as_independent_observations = FALSE,
      fixed_effects_fallback = FALSE,
      payload_fidelity_contract = payload_fidelity_contract,
      input_files = input_manifest,
      environment = environment,
      outputs = files
    )
    write_json(manifest, file.path(stage, "mixed_model_run_manifest.json"))
  })

  cat(jsonlite::toJSON(
    list(
      status = if (isTRUE(options$validate_only)) "validated" else "completed",
      output_dir = normalizePath(output_directory, mustWork = TRUE),
      analysis_unit = "payload_trial",
      segments_as_independent_observations = FALSE,
      fixed_effects_fallback = FALSE
    ),
    auto_unbox = TRUE,
    pretty = TRUE
  ))
  cat("\n")
  invisible(0L)
}

if (!identical(Sys.getenv("RANKCLOAK_MIXED_MODELS_SOURCE_ONLY"), "1")) {
  main()
}
