# This R script reads univariate numeric data from stdin,
# runs multiple modality tests from the 'multimode' package,
# and reports the p-values for each along with a conservative decision.

# Check if 'multimode' is installed. If not, install it from CRAN.
if (!requireNamespace("multimode", quietly = TRUE)) {
  install.packages("multimode", repos = "https://cloud.r-project.org/")
}
# Load the multimode package without showing startup messages
suppressMessages(library(multimode))

# Get command-line arguments passed to the script
args <- commandArgs(trailingOnly = TRUE)

# Check for at least one argument (mod0)
if (length(args) < 1) {
  stop("Usage: Rscript conservative_modality_test.R mod0 [method1 method2 ...]")
}

# Extract the number of modes under H0 (null hypothesis), typically 1
mod0 <- as.integer(args[1])

# Optional: list of methods (SI, HY, HH, CH, ACR), or use default
if (length(args) > 1) {
  methods <- args[2:length(args)]
} else {
  methods <- c("SI", "HY", "HH", "CH", "ACR")
}

# Read a column of numeric values from standard input (e.g. passed by Python)
# This is expected to be a single-column numeric input with no header
x <- scan(file("stdin"), what = numeric(), quiet = TRUE)


# Compute p-values for each method using modetest
# If a method fails (e.g. due to sample issues), return NA for that method
pvals <- sapply(methods, function(m) {
  tryCatch({
    result <- modetest(x, mod0 = mod0, method = m)  # perform the test
    result$p.value  # extract the p-value
  }, error = function(e) NA)  # return NA on failure e.g. the data is too short, or the test crashes
})

# Return only method and p-value
out_df <- data.frame(Method = methods, P_Value = pvals)
write.csv(out_df, stdout(), row.names = FALSE)
