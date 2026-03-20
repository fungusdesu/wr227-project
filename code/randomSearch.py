from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _data_dir() -> Path:
	return Path(__file__).resolve().parent.parent / 'OULAD'


def _load_data():
	data_dir = _data_dir()
	X_train = pd.read_csv(data_dir / 'X_train.csv')
	y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
	X_test = pd.read_csv(data_dir / 'X_test.csv')
	y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()
	return X_train, y_train, X_test, y_test


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
		'Gaussian Naive Bayes': (
			GaussianNB(),
			{
				'var_smoothing': np.logspace(-12, -6, 200),
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
	}
	return models


def _run_model_search(
	model_name: str,
	estimator,
	param_distributions: dict,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	n_iter: int,
	random_state: int,
):
	cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

	search = RandomizedSearchCV(
		estimator=estimator,
		param_distributions=param_distributions,
		n_iter=n_iter,
		scoring='balanced_accuracy',
		cv=cv,
		random_state=random_state,
		n_jobs=-1,
		verbose=1,
	)

	search.fit(X_train, y_train)
	best_model = search.best_estimator_
	y_pred = best_model.predict(X_test)

	result = {
		'Model': model_name,
		'Best Params': search.best_params_,
		'CV Balanced Accuracy': search.best_score_,
		'Test Accuracy': accuracy_score(y_test, y_pred),
		'Test Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
		'Test F1 (weighted)': f1_score(y_test, y_pred, average='weighted'),
	}
	return result


def main():
	random_state = 69
	n_iter = 10

	X_train, y_train, X_test, y_test = _load_data()
	configs = _search_configs(random_state=random_state)

	results = []
	for model_name, (estimator, distributions) in configs.items():
		print(f'\n--- RandomizedSearchCV: {model_name} ---')
		result = _run_model_search(
			model_name=model_name,
			estimator=estimator,
			param_distributions=distributions,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			n_iter=n_iter,
			random_state=random_state,
		)
		results.append(result)

		print(f"Best Params: {result['Best Params']}")
		print(f"CV Balanced Accuracy: {result['CV Balanced Accuracy']:.4f}")
		print(f"Test Accuracy: {result['Test Accuracy']:.4f}")
		print(f"Test Balanced Accuracy: {result['Test Balanced Accuracy']:.4f}")
		print(f"Test F1 (weighted): {result['Test F1 (weighted)']:.4f}")

	results_df = pd.DataFrame(results)
	print('\n===== Final Summary =====')
	print(results_df[['Model', 'CV Balanced Accuracy', 'Test Accuracy', 'Test Balanced Accuracy', 'Test F1 (weighted)']])

	output_path = Path(__file__).resolve().parent / 'random_search_results.csv'
	results_df.to_csv(output_path, index=False)
	print(f'\nSaved full results to: {output_path}')


if __name__ == '__main__':
	main()
