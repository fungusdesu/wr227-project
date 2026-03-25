user_lib <- file.path(Sys.getenv("HOME"), ".R", "library")
if (!user_lib %in% .libPaths()) {
  .libPaths(c(user_lib, .libPaths()))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)

enable_parallel <- tolower(Sys.getenv("WR227_ENABLE_PARALLEL", "false")) %in% c("1", "true", "yes")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    suppressWarnings(
      try(
        install.packages(pkg, repos = "https://cloud.r-project.org", lib = user_lib),
        silent = TRUE
      )
    )
  }
}

required_packages <- c("caret", "randomForest", "kernlab", "C50", "nnet", "gbm")
if (enable_parallel) {
  required_packages <- c(required_packages, "foreach", "doParallel", "doSNOW", "import")
}
for (pkg in required_packages) {
  install_if_missing(pkg)
}

if (!requireNamespace("elmNNRcpp", quietly = TRUE) && !requireNamespace("elmNN", quietly = TRUE)) {
  install_if_missing("elmNNRcpp")
}

suppressPackageStartupMessages({
  library(caret)
  if (enable_parallel && requireNamespace("doParallel", quietly = TRUE)) {
    library(doParallel)
  }
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

get_plots_dir <- function() {
  plots_dir <- file.path(script_dir, "plots")
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  plots_dir
}

load_data <- function() {
  data_dir <- get_data_dir()

  X_train <- read.csv(file.path(data_dir, "X_train.csv"), check.names = FALSE)
  y_train <- read.csv(file.path(data_dir, "y_train.csv"), check.names = FALSE)[[1]]

  y_train <- as.factor(y_train)

  list(X_train = X_train, y_train = y_train)
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

get_available_methods <- function() {
  unique(modelLookup()$model)
}

get_method_required_packages <- function(method_name) {
  info <- try(getModelInfo(method_name, regex = FALSE)[[1]], silent = TRUE)
  if (inherits(info, "try-error") || is.null(info)) {
    return(character(0))
  }

  libs <- info$library
  if (is.null(libs)) {
    return(character(0))
  }

  unique(as.character(libs))
}

ensure_method_dependencies <- function(method_name) {
  required_pkgs <- get_method_required_packages(method_name)
  if (length(required_pkgs) == 0) {
    return(TRUE)
  }

  missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    for (pkg in missing_pkgs) {
      install_if_missing(pkg)
    }
  }

  still_missing <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  length(still_missing) == 0
}

first_available_method <- function(candidates, available_methods) {
  matches <- candidates[candidates %in% available_methods]
  if (length(matches) == 0) {
    return(NULL)
  }
  matches[1]
}

safe_train <- function(train_args) {
  tryCatch(
    {
      do.call(train, train_args)
    },
    error = function(e) {
      structure(
        list(error = conditionMessage(e)),
        class = "train_error"
      )
    }
  )
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

extract_cv_balanced_accuracy_stats <- function(fit) {
  if (!"Balanced_Accuracy" %in% names(fit$results)) {
    return(list(mean = NA_real_, std = NA_real_))
  }

  std_col <- if ("Balanced_AccuracySD" %in% names(fit$results)) "Balanced_AccuracySD" else NULL

  if (is.null(fit$bestTune) || nrow(fit$bestTune) == 0) {
    best_idx <- which.max(fit$results$Balanced_Accuracy)
    cv_mean <- fit$results$Balanced_Accuracy[best_idx]
    cv_std <- if (is.null(std_col)) NA_real_ else fit$results[[std_col]][best_idx]
    return(list(mean = cv_mean, std = cv_std))
  }

  tune_cols <- names(fit$bestTune)
  matches <- rep(TRUE, nrow(fit$results))

  for (col in tune_cols) {
    matches <- matches & (fit$results[[col]] == fit$bestTune[[col]][1])
  }

  if (!any(matches)) {
    best_idx <- which.max(fit$results$Balanced_Accuracy)
    cv_mean <- fit$results$Balanced_Accuracy[best_idx]
    cv_std <- if (is.null(std_col)) NA_real_ else fit$results[[std_col]][best_idx]
    return(list(mean = cv_mean, std = cv_std))
  }

  best_idx <- which(matches)[1]
  cv_mean <- fit$results$Balanced_Accuracy[best_idx]
  cv_std <- if (is.null(std_col)) NA_real_ else fit$results[[std_col]][best_idx]
  list(mean = cv_mean, std = cv_std)
}

run_model_search <- function(
  model_name,
  method,
  X_train,
  y_train,
  tr_control,
  tune_grid,
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
      tuneGrid = tune_grid,
      metric = "Balanced_Accuracy"
    ),
    if (!is.null(pre_process)) list(preProcess = pre_process) else list(),
    extra_args
  )

  fit <- safe_train(train_args)
  if (inherits(fit, "train_error")) {
    cat(sprintf("Model failed: %s\n", fit$error))
    return(list(
      result = data.frame(
        Model = model_name,
        `Best Params` = NA_character_,
        `CV Balanced Accuracy` = NA_real_,
        `CV Balanced Accuracy Mean` = NA_real_,
        `CV Balanced Accuracy Std` = NA_real_,
        check.names = FALSE
      ),
      fit = NULL
    ))
  }

  cv_bal_acc <- extract_cv_balanced_accuracy_stats(fit)

  result <- data.frame(
    Model = model_name,
    `Best Params` = best_params_string(fit$bestTune),
    `CV Balanced Accuracy` = cv_bal_acc$mean,
    `CV Balanced Accuracy Mean` = cv_bal_acc$mean,
    `CV Balanced Accuracy Std` = cv_bal_acc$std,
    check.names = FALSE
  )

  cat(sprintf("Best Params: %s\n", result$`Best Params`[1]))
  cat(sprintf("CV Balanced Accuracy Mean: %.4f\n", result$`CV Balanced Accuracy Mean`[1]))
  cat(sprintf("CV Balanced Accuracy Std: %.4f\n", result$`CV Balanced Accuracy Std`[1]))

  list(result = result, fit = fit)
}

plot_cv_results <- function(results_df) {
  plot_path <- file.path(get_plots_dir(), "random_search_cv_r.png")

  valid_rows <- !is.na(results_df$`CV Balanced Accuracy Mean`)
  plot_df <- results_df[valid_rows, c("Model", "CV Balanced Accuracy Mean", "CV Balanced Accuracy Std")]

  if (nrow(plot_df) == 0) {
    warning("No valid CV results available to plot.")
    return(plot_path)
  }

  ordering <- order(plot_df$`CV Balanced Accuracy Mean`, decreasing = TRUE)
  plot_df <- plot_df[ordering, ]

  means <- plot_df$`CV Balanced Accuracy Mean`
  stds <- plot_df$`CV Balanced Accuracy Std`
  stds[is.na(stds)] <- 0
  labels <- plot_df$Model

  png(plot_path, width = 1400, height = 800, res = 140)
  op <- par(mar = c(9, 5, 4, 2) + 0.1)

  x_pos <- barplot(
    means,
    names.arg = labels,
    ylim = c(0, 1),
    col = "#4C78A8",
    border = NA,
    ylab = "CV Balanced Accuracy",
    main = "Random Search CV Performance (Mean ± Std)",
    las = 2
  )

  arrows(
    x0 = x_pos,
    y0 = pmax(0, means - stds),
    x1 = x_pos,
    y1 = pmin(1, means + stds),
    angle = 90,
    code = 3,
    length = 0.04,
    col = "#333333",
    lwd = 1.2
  )

  par(op)
  dev.off()

  plot_path
}

build_elm_no_kernel_grid <- function() {
  neuron_values <- unique(round(seq(3, 200, length.out = 20)))
  activation_values <- c("sin", "sign", "sig", "hardlim", "tribas", "radbas")

  expand.grid(
    nhidden = neuron_values,
    actfun = activation_values,
    stringsAsFactors = FALSE
  )
}

build_elm_kernel_grid <- function(method_name) {
  params <- subset(modelLookup(), model == method_name)$parameter
  if (length(params) < 2) {
    return(NULL)
  }

  reg_values <- 2^(-5:14)
  spread_values <- 2^(-16:8)
  tune_values <- vector("list", length(params))
  names(tune_values) <- params

  for (param_name in params) {
    lower_name <- tolower(param_name)
    if (grepl("(sigma|gamma|spread)", lower_name)) {
      tune_values[[param_name]] <- spread_values
    } else if (grepl("(lambda|c|cost|reg|alpha)", lower_name)) {
      tune_values[[param_name]] <- reg_values
    }
  }

  empty_slots <- vapply(tune_values, is.null, logical(1))
  if (any(empty_slots)) {
    unknown_params <- names(tune_values)[empty_slots]
    for (i in seq_along(unknown_params)) {
      param <- unknown_params[i]
      tune_values[[param]] <- if (i == 1) reg_values else spread_values
    }
  }

  do.call(expand.grid, c(tune_values, stringsAsFactors = FALSE))
}

build_model_configs <- function(X_train) {
  p <- ncol(X_train)
  mtry_sqrt <- max(1L, floor(sqrt(p)))
  available_methods <- get_available_methods()

  elm_no_kernel_method <- first_available_method(c("elm", "elmNN"), available_methods)
  elm_kernel_method <- first_available_method(c("elm_kernel_m", "elm_kernel", "elmKernel", "elm_km"), available_methods)

  model_configs <- list(
    list(
      name = "rf t",
      method = "rf",
      tune_grid = data.frame(mtry = seq(2, 29, by = 3)),
      pre_process = NULL,
      extra_args = list(ntree = 500)
    ),
    list(
      name = "rforest R",
      method = "rf",
      tune_grid = data.frame(mtry = mtry_sqrt),
      pre_process = NULL,
      extra_args = list(ntree = 500)
    ),
    list(
      name = "svm C (Gaussian kernel)",
      method = "svmRadial",
      tune_grid = expand.grid(
        sigma = 2^(-16:8),
        C = 2^(-5:14)
      ),
      pre_process = c("center", "scale"),
      extra_args = list()
    ),
    list(
      name = "svmPoly t",
      method = "svmPoly",
      tune_grid = expand.grid(
        degree = c(1, 2, 3),
        scale = c(0.001, 0.01, 0.1),
        C = c(0.25, 0.5, 1)
      ),
      pre_process = c("center", "scale"),
      extra_args = list(kpar = list(offset = 1))
    ),
    list(
      name = "svmRadial t",
      method = "svmRadial",
      tune_grid = expand.grid(
        sigma = 10^(-2:2),
        C = 2^(-2:2)
      ),
      pre_process = c("center", "scale"),
      extra_args = list()
    ),
    list(
      name = "svmRadialCost t",
      method = "svmRadialCost",
      tune_grid = expand.grid(C = 2^(-2:2)),
      pre_process = c("center", "scale"),
      extra_args = list()
    ),
    list(
      name = "C5.0 t",
      method = "C5.0",
      tune_grid = expand.grid(
        trials = c(1, 10, 20),
        model = "tree",
        winnow = c(FALSE, TRUE)
      ),
      pre_process = NULL,
      extra_args = list()
    ),
    list(
      name = "avNNet t",
      method = "avNNet",
      tune_grid = expand.grid(
        size = c(1, 3, 5),
        decay = c(0, 0.1, 1e-4),
        bag = FALSE
      ),
      pre_process = c("center", "scale"),
      extra_args = list(trace = FALSE, linout = FALSE, MaxNWts = 100000, repeats = 5)
    )
  )

  if (enable_parallel) {
    model_configs <- c(
      list(
        list(
          name = "parRF t",
          method = "parRF",
          tune_grid = data.frame(mtry = seq(2, 8, by = 2)),
          pre_process = NULL,
          extra_args = list(ntree = 500)
        )
      ),
      model_configs
    )
  }

  if (!is.null(elm_no_kernel_method)) {
    model_configs[[length(model_configs) + 1]] <- list(
      name = "elm m (No kernel)",
      method = elm_no_kernel_method,
      tune_grid = build_elm_no_kernel_grid(),
      pre_process = NULL,
      extra_args = list()
    )
  } else {
    message("Skipping 'elm m (No kernel)': no caret method found ('elm'/'elmNN').")
  }

  if (!is.null(elm_kernel_method)) {
    model_configs[[length(model_configs) + 1]] <- list(
      name = "elm kernel m (Gaussian kernel)",
      method = elm_kernel_method,
      tune_grid = build_elm_kernel_grid(elm_kernel_method),
      pre_process = c("center", "scale"),
      extra_args = list()
    )
  } else {
    message("Skipping 'elm kernel m (Gaussian kernel)': no caret kernel ELM method found.")
  }

  available_methods <- get_available_methods()
  Filter(function(cfg) {
    if (!(cfg$method %in% available_methods)) {
      message(sprintf("Skipping '%s': caret method '%s' is unavailable.", cfg$name, cfg$method))
      return(FALSE)
    }

    if (!ensure_method_dependencies(cfg$method)) {
      message(sprintf("Skipping '%s': missing required package(s) for method '%s'.", cfg$name, cfg$method))
      return(FALSE)
    }

    if (cfg$method == "parRF" && (!enable_parallel || !requireNamespace("import", quietly = TRUE))) {
      message("Skipping 'parRF t': parallel mode or package 'import' unavailable.")
      return(FALSE)
    }

    TRUE
  }, model_configs)
}

main <- function() {
  random_state <- 69

  data <- load_data()

  tr_control <- trainControl(
    method = "cv",
    number = 5,
    summaryFunction = custom_summary,
    classProbs = FALSE,
    savePredictions = "final",
    allowParallel = FALSE
  )

  model_configs <- build_model_configs(data$X_train)

  search_outputs <- lapply(model_configs, function(cfg) {
    run_model_search(
      model_name = cfg$name,
      method = cfg$method,
      X_train = data$X_train,
      y_train = data$y_train,
      tr_control = tr_control,
      tune_grid = cfg$tune_grid,
      random_state = random_state,
      pre_process = cfg$pre_process,
      extra_args = cfg$extra_args
    )
  })

  results_df <- do.call(rbind, lapply(search_outputs, function(x) x$result))
  refit_models <- lapply(search_outputs, function(x) x$fit)
  names(refit_models) <- vapply(model_configs, function(cfg) cfg$name, character(1))
  refit_models <- Filter(Negate(is.null), refit_models)

  output_path <- file.path(script_dir, "random_search_results_r.csv")
  write.csv(results_df, output_path, row.names = FALSE)
  plot_path <- plot_cv_results(results_df)
  refit_path <- file.path(script_dir, "refit_models_r.rds")
  saveRDS(refit_models, refit_path)

  cat("\n===== Final Summary =====\n")
  print(results_df[, c("Model", "CV Balanced Accuracy Mean", "CV Balanced Accuracy Std")])
  cat(sprintf("\nSaved full results to: %s\n", output_path))
  cat(sprintf("Saved plot to: %s\n", plot_path))
  cat(sprintf("Saved refit models to: %s\n", refit_path))
}

main()
