from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from skelm import ELMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def _data_dir() -> Path:
	return Path(__file__).resolve().parent.parent / 'OULAD'


def _plots_dir() -> Path:
	plots_dir = Path(__file__).resolve().parent / 'plots'
	plots_dir.mkdir(parents=True, exist_ok=True)
	return plots_dir


def _refit_models_path() -> Path:
	return Path(__file__).resolve().parent / 'refit_models_py.joblib'


def _picklable_refit_models(refit_models: dict):
	picklable = {}
	skipped = []

	for model_name, estimator in refit_models.items():
		try:
			pickle.dumps(estimator)
			picklable[model_name] = estimator
		except Exception:
			skipped.append(model_name)

	return picklable, skipped


def _load_data():
	data_dir = _data_dir()
	X_train = pd.read_csv(data_dir / 'X_train.csv')
	y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
	return X_train, y_train


def _search_configs(random_state: int):
	models = {
		'Random Forest': (
			RandomForestClassifier(random_state=random_state, n_jobs=-1),
			{
				'n_estimators': [100, 200, 300, 500, 800],
				'max_depth': [None, 5, 10, 20, 30, 50],
				'min_samples_split': [2, 5, 10, 20],
				'min_samples_leaf': [1, 2, 4, 8],
				'max_features': ['sqrt', 'log2', None],
				'bootstrap': [True, False],
			},
		),
		'Decision Tree': (
			DecisionTreeClassifier(random_state=random_state),
			{
				'criterion': ['gini', 'entropy', 'log_loss'],
				'splitter': ['best', 'random'],
				'max_depth': [None, 5, 10, 20, 30, 50],
				'min_samples_split': [2, 5, 10, 20],
				'min_samples_leaf': [1, 2, 4, 8],
				'max_features': ['sqrt', 'log2', None],
			},
		),
		'Support Vector Machine': (
			Pipeline(
				[
					('scaler', StandardScaler()),
					('svc', SVC(random_state=random_state)),
				]
			),
			{
				'svc__C': np.logspace(-2, 2, 20),
				'svc__gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1, 1.0],
				'svc__kernel': ['rbf', 'linear', 'poly'],
				'svc__class_weight': [None, 'balanced'],
			},
		),
		'Gradient Boosting': (
			GradientBoostingClassifier(random_state=random_state),
			{
				'n_estimators': [25, 50, 75, 100],
				'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
				'max_depth': [2, 3, 5, 7],
				'min_samples_split': [2, 5, 10, 20],
				'min_samples_leaf': [1, 2, 4, 8],
				'subsample': [0.6, 0.8, 1.0],
				'max_features': ['sqrt', 'log2', None],
			},
		),
		'K-Nearest Neighbors': (
			Pipeline(
				[
					('scaler', StandardScaler()),
					('knn', KNeighborsClassifier()),
				]
			),
			{
				'knn__n_neighbors': list(range(3, 52, 2)),
				'knn__weights': ['uniform', 'distance'],
				'knn__p': [1, 2],
				'knn__metric': ['minkowski', 'euclidean', 'manhattan'],
				'knn__leaf_size': [10, 20, 30, 40, 50],
			},
		),
		'Extreme Learning Machine': (
			Pipeline(
				[
					('scaler', StandardScaler()),
					('elm', ELMClassifier(random_state=random_state)),
				]
			),
			{
				'elm__n_neurons': [100, 300, 500, 700, 1000],
				'elm__ufunc': ['tanh', 'sigm', 'relu'],
				'elm__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
			},
		),
		'Multi-layer Perceptron': (
			Pipeline(
				[
					('scaler', StandardScaler()),
					('mlpc', MLPClassifier(random_state=random_state)),
				]
			),
			[
				{
					'mlpc__solver': ['adam'],
					'mlpc__hidden_layer_sizes': [(32,), (50,), (100,), (50, 50)],
					'mlpc__activation': ['relu', 'tanh'],
					'mlpc__alpha': [1e-4, 1e-3, 1e-2],
					'mlpc__learning_rate_init': [1e-4, 1e-3, 1e-2],
					'mlpc__max_iter': [1000, 1500, 2000],
					'mlpc__early_stopping': [True],
					'mlpc__n_iter_no_change': [10, 20],
				},
				{
					'mlpc__solver': ['sgd'],
					'mlpc__hidden_layer_sizes': [(32,), (50,), (100,)],
					'mlpc__activation': ['relu', 'tanh'],
					'mlpc__alpha': [1e-4, 1e-3, 1e-2],
					'mlpc__learning_rate': ['constant', 'adaptive'],
					'mlpc__learning_rate_init': [1e-4, 1e-3, 1e-2],
					'mlpc__max_iter': [1000, 1500, 2000],
					'mlpc__early_stopping': [True],
					'mlpc__n_iter_no_change': [10, 20],
				},
				{
					'mlpc__solver': ['lbfgs'],
					'mlpc__hidden_layer_sizes': [(32,), (50,), (100,)],
					'mlpc__activation': ['relu', 'tanh'],
					'mlpc__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
					'mlpc__max_iter': [500, 1000, 1500],
					'mlpc__max_fun': [15000, 30000],
				},
			],
		),
	}
	return models



def _run_model_search(
	model_name: str,
	estimator,
	param_distributions: dict,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	n_iter: int,
	random_state: int,
):
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

	search = RandomizedSearchCV(
		estimator=estimator,
		param_distributions=param_distributions,
		n_iter=n_iter,
		scoring='balanced_accuracy',
		refit=True,
		cv=cv,
		random_state=random_state,
		n_jobs=-1,
		verbose=1,
	)

	search.fit(X_train, y_train)
	cv_mean = float(search.cv_results_['mean_test_score'][search.best_index_])
	cv_std = float(search.cv_results_['std_test_score'][search.best_index_])

	result = {
		'Model': model_name,
		'Best Params': search.best_params_,
		'CV Balanced Accuracy': cv_mean,
		'CV Balanced Accuracy Mean': cv_mean,
		'CV Balanced Accuracy Std': cv_std,
	}
	return result, search.best_estimator_


def _save_cv_plot(results_df: pd.DataFrame) -> Path:
	plot_path = _plots_dir() / 'random_search_cv_py.png'

	plot_df = results_df.dropna(subset=['CV Balanced Accuracy Mean']).copy()
	plot_df = plot_df.sort_values('CV Balanced Accuracy Mean', ascending=False)

	fig, ax = plt.subplots(figsize=(12, 7))
	ax.barh(
		plot_df['Model'],
		plot_df['CV Balanced Accuracy Mean'],
		xerr=plot_df['CV Balanced Accuracy Std'].fillna(0.0),
		color='#4C78A8',
		ecolor='#333333',
		capsize=4,
	)
	ax.invert_yaxis()
	ax.set_xlim(0, 1)
	ax.set_xlabel('CV Balanced Accuracy')
	ax.set_title('Random Search CV Performance (Mean ± Std)')
	ax.grid(axis='x', linestyle='--', alpha=0.4)
	fig.tight_layout()
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)

	return plot_path


def main():
	random_state = 69
	n_iter = 10

	X_train, y_train = _load_data()
	configs = _search_configs(random_state=random_state)

	results = []
	refit_models = {}
	for model_name, (estimator, distributions) in configs.items():
		print(f'\n--- RandomizedSearchCV: {model_name} ---')
		result, best_estimator = _run_model_search(
			model_name=model_name,
			estimator=estimator,
			param_distributions=distributions,
			X_train=X_train,
			y_train=y_train,
			n_iter=n_iter,
			random_state=random_state,
		)

		print(f"Best Params: {result['Best Params']}")
		print(f"CV Balanced Accuracy Mean: {result['CV Balanced Accuracy Mean']:.4f}")
		print(f"CV Balanced Accuracy Std: {result['CV Balanced Accuracy Std']:.4f}")
		results.append(result)
		refit_models[model_name] = best_estimator

	results_df = pd.DataFrame(results)
	print('\n===== Final Summary =====')
	print(results_df[['Model', 'CV Balanced Accuracy Mean', 'CV Balanced Accuracy Std']])

	output_path = Path(__file__).resolve().parent / 'random_search_results_py.csv'
	results_df.to_csv(output_path, index=False)
	plot_path = _save_cv_plot(results_df)
	serializable_models, skipped_models = _picklable_refit_models(refit_models)
	refit_path = _refit_models_path()
	joblib.dump(serializable_models, refit_path)
	print(f'\nSaved full results to: {output_path}')
	print(f'Saved plot to: {plot_path}')
	print(f'Saved refit models to: {refit_path}')
	if skipped_models:
		print(f'Skipped non-picklable refit models: {skipped_models}')


if __name__ == '__main__':
	main()
