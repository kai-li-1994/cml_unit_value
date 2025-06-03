# This R script reads univariate numeric data from stdin,
# runs SI and HH modality tests using the 'multimode' package,
# and reports the p-values along with basic decisions.

if (!requireNamespace("multimode", quietly = TRUE)) {
  install.packages("multimode", repos = "https://cloud.r-project.org/")
}
suppressMessages(library(multimode))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript uv_modality_test.R mod0 [method1 method2 ...]")
}

mod0 <- as.integer(args[1])

# Only support SI and HH methods now
if (length(args) > 1) {
  methods <- args[2:length(args)]
} else {
  methods <- c("SI", "HH")
}

x <- scan(file("stdin"), what = numeric(), quiet = TRUE)

pvals <- sapply(methods, function(m) {
  tryCatch({
    result <- modetest(x, mod0 = mod0, method = m)
    result$p.value
  }, error = function(e) NA)
})

out_df <- data.frame(Method = methods, P_Value = pvals)
write.csv(out_df, stdout(), row.names = FALSE)