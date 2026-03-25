user_lib <- file.path(Sys.getenv("HOME"), ".R", "library")
if (!user_lib %in% .libPaths()) {
  .libPaths(c(user_lib, .libPaths()))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", lib = user_lib)
  }
}

required_packages <- c("caret", "randomForest", "kernlab", "C50", "nnet", "gbm")
for (pkg in required_packages) {
  install_if_missing(pkg)
}

if (!requireNamespace("elmNNRcpp", quietly = TRUE) && !requireNamespace("elmNN", quietly = TRUE)) {
  install.packages("elmNNRcpp", repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(caret)
})

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }

  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  }

  getwd()
}

script_dir <- get_script_dir()
model_data <- NULL

get_data_dir <- function() {
  normalizePath(file.path(script_dir, "..", "OULAD"), mustWork = TRUE)
}

get_plots_dir <- function() {
  plots_dir <- file.path(script_dir, "plots")
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  plots_dir
}

load_data <- function() {
  data_dir <- get_data_dir()

  X_train <- read.csv(file.path(data_dir, "X_train.csv"), check.names = FALSE)
  y_train <- read.csv(file.path(data_dir, "y_train.csv"), check.names = FALSE)[[1]]
  X_test <- read.csv(file.path(data_dir, "X_test.csv"), check.names = FALSE)
  y_test <- read.csv(file.path(data_dir, "y_test.csv"), check.names = FALSE)[[1]]

  y_train <- as.factor(y_train)
  y_test <- factor(y_test, levels = levels(y_train))

  list(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)
}

weighted_f1_score <- function(cm) {
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)

  precision[is.na(precision)] <- 0
  recall[is.na(recall)] <- 0

  f1_per_class <- ifelse(
    (precision + recall) == 0,
    0,
    2 * precision * recall / (precision + recall)
  )

  supports <- rowSums(cm)
  sum(f1_per_class * supports) / sum(supports)
}

balanced_accuracy <- function(cm) {
  recalls <- diag(cm) / rowSums(cm)
  recalls[is.na(recalls)] <- 0
  mean(recalls)
}

evaluate_model <- function(model_name, fitted_model, X_test, y_test) {
  y_pred <- predict(fitted_model, X_test)
  y_pred <- factor(y_pred, levels = levels(y_test))

  cm <- table(y_test, y_pred)
  test_accuracy <- sum(diag(cm)) / sum(cm)
  test_bal_acc <- balanced_accuracy(cm)
  test_f1_weighted <- weighted_f1_score(cm)

  cat(sprintf("%s Test Accuracy: %.4f\n", model_name, test_accuracy))
  cat(sprintf("%s Test Balanced Accuracy: %.4f\n", model_name, test_bal_acc))
  cat(sprintf("%s Test F1 Score: %.4f\n\n", model_name, test_f1_weighted))

  data.frame(
    Model = model_name,
    `Test Accuracy` = test_accuracy,
    `Test Balanced Accuracy` = test_bal_acc,
    `Test F1 Score` = test_f1_weighted,
    check.names = FALSE
  )
}

train_models <- function() {
  if (is.null(model_data)) {
    stop("Training data not initialized.")
  }

  X_train <- model_data$X_train
  y_train <- model_data$y_train

  set.seed(69)
  control <- trainControl(method = "cv", number = 3)

  available_methods <- unique(modelLookup()$model)
  elm_method <- if ("elm" %in% available_methods) {
    "elm"
  } else if ("elmNN" %in% available_methods) {
    "elmNN"
  } else {
    stop("ELM method not available in caret. Install package 'elmNNRcpp' or 'elmNN'.")
  }

  models <- list(
    `Random Forest` = train(
      x = X_train,
      y = y_train,
      method = "rf",
      trControl = control,
      tuneGrid = data.frame(mtry = 10),
      ntree = 500
    ),
    `Support Vector Machine` = train(
      x = X_train,
      y = y_train,
      method = "svmRadial",
      trControl = control,
      tuneGrid = data.frame(sigma = 0.01, C = 1),
      preProcess = c("center", "scale")
    ),
    ELM = train(
      x = X_train,
      y = y_train,
      method = elm_method,
      trControl = control,
      tuneGrid = data.frame(nhidden = 100, actfun = "sig")
    ),
    `C5.0` = train(
      x = X_train,
      y = y_train,
      method = "C5.0",
      trControl = control,
      tuneGrid = data.frame(trials = 10, model = "tree", winnow = FALSE)
    ),
    avNNet = train(
      x = X_train,
      y = y_train,
      method = "avNNet",
      trControl = control,
      tuneGrid = data.frame(size = 5, decay = 0.0001, bag = FALSE),
      preProcess = c("center", "scale"),
      trace = FALSE,
      linout = FALSE,
      MaxNWts = 100000,
      repeats = 5
    ),
    `K-Nearest Neighbors` = train(
      x = X_train,
      y = y_train,
      method = "knn",
      trControl = control,
      tuneGrid = data.frame(k = 7),
      preProcess = c("center", "scale")
    ),
    `Gradient Boosting` = train(
      x = X_train,
      y = y_train,
      method = "gbm",
      trControl = control,
      tuneGrid = data.frame(n.trees = 100, interaction.depth = 5, shrinkage = 0.1, n.minobsinnode = 10),
      verbose = FALSE
    )
  )

  models
}

plot_results <- function(results_df) {
  plots_dir <- get_plots_dir()
  output_path <- file.path(plots_dir, "performance_r.png")

  metric_matrix <- as.matrix(results_df[, c("Test Accuracy", "Test Balanced Accuracy", "Test F1 Score")])
  rownames(metric_matrix) <- results_df$Model

  png(output_path, width = 1400, height = 800, res = 140)
  op <- par(mar = c(9, 5, 4, 2) + 0.1)

  barplot(
    t(metric_matrix),
    beside = TRUE,
    ylim = c(0, 1),
    col = c("#4C78A8", "#F58518", "#54A24B"),
    names.arg = results_df$Model,
    las = 2,
    ylab = "Score",
    main = "Model Performance Comparison"
  )

  legend(
    "topright",
    legend = c("Test Accuracy", "Test Balanced Accuracy", "Test F1 Score"),
    fill = c("#4C78A8", "#F58518", "#54A24B"),
    bty = "n"
  )

  par(op)
  dev.off()

  cat(sprintf("Plot saved to: %s\n", output_path))
}

main <- function() {
  model_data <<- load_data()
  refit_path <- file.path(script_dir, "refit_models_r.rds")
  if (!file.exists(refit_path)) {
    stop(sprintf("Refit models not found at: %s. Run randomSearch.r first.", refit_path))
  }

  models <- readRDS(refit_path)

  results_list <- lapply(names(models), function(model_name) {
    evaluate_model(model_name, models[[model_name]], model_data$X_test, model_data$y_test)
  })

  results_df <- do.call(rbind, results_list)

  csv_path <- file.path(script_dir, "model_results_r.csv")
  write.csv(results_df, csv_path, row.names = FALSE)

  cat("===== Final Summary =====\n")
  print(results_df)
  cat(sprintf("Saved full results to: %s\n", csv_path))

  plot_results(results_df)
}

main()
