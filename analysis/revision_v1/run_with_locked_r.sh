#!/usr/bin/env bash
set -euo pipefail

launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${launcher_dir}/../.." && pwd)"
project_library="${repo_root}/.r_libs/revision_v1"
existing_user_library="/home/meow/R/x86_64-pc-linux-gnu-library/4.4"
rscript_binary="${RANKCLOAK_RSCRIPT:-/usr/bin/Rscript}"

if [[ ! -d "${project_library}" || ! -d "${existing_user_library}" ]]; then
  echo "A declared locked R library is unavailable; no automatic install is permitted." >&2
  exit 2
fi
export R_LIBS_USER="${project_library}:${existing_user_library}"
export R_ENVIRON_USER=/dev/null
export R_PROFILE_USER=/dev/null

if [[ ! -x "${rscript_binary}" ]]; then
  echo "Locked Rscript is unavailable: ${rscript_binary}" >&2
  exit 2
fi

"${rscript_binary}" --vanilla -e '
expected_r <- "4.4.2"
if (paste(R.version$major, R.version$minor, sep=".") != expected_r) {
  stop(sprintf("R version mismatch: expected %s, found %s", expected_r, R.version.string))
}
libs <- strsplit(Sys.getenv("R_LIBS_USER"), .Platform$path.sep, fixed=TRUE)[[1L]]
libs <- normalizePath(libs, mustWork=TRUE)
expected <- c(lme4="2.0.6", ordinal="2026.7.26", emmeans="1.10.5", jsonlite="1.8.9")
expected_library <- c(lme4=libs[[1L]], ordinal=libs[[1L]], emmeans=libs[[2L]], jsonlite=libs[[2L]])
for (pkg in names(expected)) {
  location <- find.package(pkg, quiet=TRUE)
  if (!nzchar(location)) {
    stop(sprintf("locked package missing: %s %s", pkg, expected[[pkg]]))
  }
  found <- as.character(utils::packageVersion(pkg))
  if (utils::compareVersion(found, expected[[pkg]]) != 0L) {
    stop(sprintf("locked package mismatch for %s: expected %s, found %s", pkg, expected[[pkg]], found))
  }
  resolved <- normalizePath(location, mustWork=TRUE)
  if (!startsWith(resolved, paste0(expected_library[[pkg]], .Platform$file.sep))) {
    stop(sprintf("package %s resolved from an undeclared library: %s", pkg, resolved))
  }
  cat(sprintf("locked package %s %s => %s\n", pkg, found, resolved), file=stderr())
}
'

exec "${rscript_binary}" --vanilla "$@"
