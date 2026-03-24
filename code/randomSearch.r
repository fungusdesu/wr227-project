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
  install.packages("elmNNRcpp", repos = "https://cloud.r-project.org", lib = user_lib)
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

get_data_dir <- function() {
  normalizePath(file.path(script_dir, "..", "OULAD"), mustWork = TRUE)
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

custom_summary <- function(data, lev = NULL, model = NULL) {
  obs <- factor(data$obs, levels = lev)
  pred <- factor(data$pred, levels = lev)
  cm <- table(obs, pred)

  acc <- sum(diag(cm)) / sum(cm)
  bal_acc <- balanced_accuracy(cm)
  f1_weighted <- weighted_f1_score(cm)

  c(
    Accuracy = acc,
    Balanced_Accuracy = bal_acc,
    F1_Weighted = f1_weighted
  )
}

get_elm_method <- function() {
  available_methods <- unique(modelLookup()$model)
  if ("elm" %in% available_methods) {
    return("elm")
  }
  if ("elmNN" %in% available_methods) {
    return("elmNN")
  }
  stop("ELM method not available in caret. Install package 'elmNNRcpp' or 'elmNN'.")
}

best_params_string <- function(best_tune_df) {
  if (is.null(best_tune_df) || nrow(best_tune_df) == 0) {
    return(NA_character_)
  }

  paste(
    vapply(
      names(best_tune_df),
      function(col) {
        value <- as.character(best_tune_df[[col]][1])
        paste0(col, "=", value)
      },
      character(1)
    ),
    collapse = ", "
  )
}

extract_cv_balanced_accuracy <- function(fit) {
  if (!"Balanced_Accuracy" %in% names(fit$results)) {
    return(NA_real_)
  }

  if (is.null(fit$bestTune) || nrow(fit$bestTune) == 0) {
    return(max(fit$results$Balanced_Accuracy, na.rm = TRUE))
  }

  tune_cols <- names(fit$bestTune)
  matches <- rep(TRUE, nrow(fit$results))

  for (col in tune_cols) {
    matches <- matches & (fit$results[[col]] == fit$bestTune[[col]][1])
  }

  if (!any(matches)) {
    return(max(fit$results$Balanced_Accuracy, na.rm = TRUE))
  }

  fit$results$Balanced_Accuracy[which(matches)[1]]
}

run_model_search <- function(
  model_name,
  method,
  X_train,
  y_train,
  X_test,
  y_test,
  tr_control,
  n_iter,
  random_state,
  pre_process = NULL,
  extra_args = list()
) {
  cat(sprintf("\n--- Random Search: %s ---\n", model_name))

  set.seed(random_state)
  train_args <- c(
    list(
      x = X_train,
      y = y_train,
      method = method,
      trControl = tr_control,
      tuneLength = n_iter,
      metric = "Balanced_Accuracy"
    ),
    if (!is.null(pre_process)) list(preProcess = pre_process) else list(),
    extra_args
  )

  fit <- do.call(train, train_args)
  y_pred <- predict(fit, X_test)
  y_pred <- factor(y_pred, levels = levels(y_test))

  cm <- table(y_test, y_pred)
  test_accuracy <- sum(diag(cm)) / sum(cm)
  test_bal_acc <- balanced_accuracy(cm)
  test_f1_weighted <- weighted_f1_score(cm)
  cv_bal_acc <- extract_cv_balanced_accuracy(fit)

  result <- data.frame(
    Model = model_name,
    `Best Params` = best_params_string(fit$bestTune),
    `CV Balanced Accuracy` = cv_bal_acc,
    `Test Accuracy` = test_accuracy,
    `Test Balanced Accuracy` = test_bal_acc,
    `Test F1 (weighted)` = test_f1_weighted,
    check.names = FALSE
  )

  cat(sprintf("Best Params: %s\n", result$`Best Params`[1]))
  cat(sprintf("CV Balanced Accuracy: %.4f\n", result$`CV Balanced Accuracy`[1]))
  cat(sprintf("Test Accuracy: %.4f\n", result$`Test Accuracy`[1]))
  cat(sprintf("Test Balanced Accuracy: %.4f\n", result$`Test Balanced Accuracy`[1]))
  cat(sprintf("Test F1 (weighted): %.4f\n", result$`Test F1 (weighted)`[1]))

  result
}

main <- function() {
  random_state <- 69
  n_iter <- 10

  data <- load_data()

  tr_control <- trainControl(
    method = "cv",
    number = 3,
    search = "random",
    summaryFunction = custom_summary,
    classProbs = FALSE,
    savePredictions = "final"
  )

  elm_method <- get_elm_method()

  model_configs <- list(
    list(
      name = "Random Forest",
      method = "rf",
      pre_process = NULL,
      extra_args = list(ntree = 300)
    ),
    list(
      name = "Support Vector Machine",
      method = "svmRadial",
      pre_process = c("center", "scale"),
      extra_args = list()
    ),
    list(
      name = "ELM",
      method = elm_method,
      pre_process = NULL,
      extra_args = list()
    ),
    list(
      name = "C5.0",
      method = "C5.0",
      pre_process = NULL,
      extra_args = list()
    ),
    list(
      name = "avNNet",
      method = "avNNet",
      pre_process = c("center", "scale"),
      extra_args = list(trace = FALSE, linout = FALSE, MaxNWts = 100000)
    ),
    list(
      name = "K-Nearest Neighbors",
      method = "knn",
      pre_process = c("center", "scale"),
      extra_args = list()
    ),
    list(
      name = "Gradient Boosting",
      method = "gbm",
      pre_process = NULL,
      extra_args = list(verbose = FALSE)
    )
  )

  results_list <- lapply(model_configs, function(cfg) {
    run_model_search(
      model_name = cfg$name,
      method = cfg$method,
      X_train = data$X_train,
      y_train = data$y_train,
      X_test = data$X_test,
      y_test = data$y_test,
      tr_control = tr_control,
      n_iter = n_iter,
      random_state = random_state,
      pre_process = cfg$pre_process,
      extra_args = cfg$extra_args
    )
  })

  results_df <- do.call(rbind, results_list)

  output_path <- file.path(script_dir, "random_search_results_r.csv")
  write.csv(results_df, output_path, row.names = FALSE)

  cat("\n===== Final Summary =====\n")
  print(results_df)
  cat(sprintf("\nSaved full results to: %s\n", output_path))
}

main()
