import ast
import re
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import dask.array as da
from skelm import ELMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import pandas as pd
import pathlib as path
import matplotlib.pyplot as plt
import numpy as np

_ = da


def _data_dir():
    return path.Path(__file__).resolve().parent.parent / 'OULAD'


def _load_train_data():
    data_dir = _data_dir()
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    return X_train, y_train


def _plots_dir():
    plots_dir = path.Path(__file__).resolve().parent / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _random_search_results_path():
    return path.Path(__file__).resolve().parent / 'random_search_results_py.csv'


def _refit_models_path():
    return path.Path(__file__).resolve().parent / 'refit_models_py.joblib'


def _model_results_path():
    return path.Path(__file__).resolve().parent / 'model_performance_results_py.csv'


def _load_refit_models():
    refit_path = _refit_models_path()
    if not refit_path.exists():
        raise FileNotFoundError(
            f'Refit models not found at: {refit_path}. Run randomSearch.py first.'
        )
    loaded = joblib.load(refit_path)
    if not isinstance(loaded, dict):
        raise ValueError(f'Invalid refit model artifact at: {refit_path}')
    return loaded


def _safe_parse_best_params(value):
    if pd.isna(value):
        return {}

    raw = str(value).strip()
    if not raw:
        return {}

    sanitized = re.sub(r'np\.float\d+\(([^\)]*)\)', r'\1', raw)
    sanitized = re.sub(r'np\.int\d+\(([^\)]*)\)', r'\1', sanitized)

    try:
        parsed = ast.literal_eval(sanitized)
    except (ValueError, SyntaxError):
        return {}

    if not isinstance(parsed, dict):
        return {}
    return parsed


def _load_tuned_params():
    results_path = _random_search_results_path()
    if not results_path.exists():
        print(f'No random search results found at: {results_path}. Using fallback parameters.')
        return {}

    df = pd.read_csv(results_path)
    if 'Model' not in df.columns or 'Best Params' not in df.columns:
        print(f'Invalid random search results format in: {results_path}. Using fallback parameters.')
        return {}

    tuned = {}
    for _, row in df.iterrows():
        model_name = str(row['Model'])
        tuned[model_name] = _safe_parse_best_params(row['Best Params'])

    return tuned


def _pick_model_params(tuned_params, model_name, aliases=None):
    if aliases is None:
        aliases = []

    for candidate in [model_name, *aliases]:
        if candidate in tuned_params:
            return tuned_params[candidate]
    return {}


def _strip_prefix(params, prefix):
    extracted = {}
    for key, value in params.items():
        if key.startswith(prefix):
            extracted[key[len(prefix):]] = value
    return extracted


def _filter_params(estimator, params):
    valid_keys = set(estimator.get_params().keys())
    return {key: value for key, value in params.items() if key in valid_keys}


def _build_elm_fallback(tuned_params):
    params = _pick_model_params(tuned_params, 'Extreme Learning Machine', aliases=['ELM Classifier'])
    fallback_defaults = {
        'elm__ufunc': 'relu',
        'elm__n_neurons': 1000,
        'elm__alpha': 0.0001,
    }
    combined_params = {**fallback_defaults, **params}

    elm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elm', ELMClassifier(random_state=69)),
    ])
    elm_pipeline.set_params(**_filter_params(elm_pipeline, combined_params))
    return elm_pipeline


def evaluate_model(model_name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')

    print(f'{model_name} Test Accuracy: {test_accuracy:.4f}')
    print(f'{model_name} Balanced Accuracy: {test_balanced_accuracy:.4f}')
    print(f'{model_name} Macro F1: {test_f1:.4f}')
    print()

    results.append({
        'Model': model_name,
        'Test Accuracy': test_accuracy,
        'Balanced Accuracy': test_balanced_accuracy,
        'Macro F1': test_f1
    })

def decision_tree(X_test, y_test, results, tuned_params):
    model = DecisionTreeClassifier(random_state=69)
    params = _pick_model_params(tuned_params, 'Decision Tree')
    model.set_params(**_filter_params(model, params))
    evaluate_model('Decision Tree', model, X_test, y_test, results)


def random_forest(X_test, y_test, results, tuned_params):
    model = RandomForestClassifier(random_state=69)
    params = _pick_model_params(tuned_params, 'Random Forest')
    model.set_params(**_filter_params(model, params))
    evaluate_model('Random Forest', model, X_test, y_test, results)


def gradient_boosting(X_test, y_test, results, tuned_params):
    model = GradientBoostingClassifier()
    params = _pick_model_params(tuned_params, 'Gradient Boosting')
    model.set_params(**_filter_params(model, params))
    evaluate_model('Gradient Boosting', model, X_test, y_test, results)


def support_vector_machine(X_test, y_test, results, tuned_params):
    params = _pick_model_params(tuned_params, 'Support Vector Machine')
    svm_params = _strip_prefix(params, 'svc__')
    svm = SVC(random_state=0)
    svm.set_params(**_filter_params(svm, svm_params))

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', svm)
    ])
    evaluate_model('Support Vector Machine', model, X_test, y_test, results)


def k_nearest_neighbors(X_test, y_test, results, tuned_params):
    params = _pick_model_params(tuned_params, 'K-Nearest Neighbors')
    knn_params = _strip_prefix(params, 'knn__')
    knn = KNeighborsClassifier()
    knn.set_params(**_filter_params(knn, knn_params))

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', knn)
    ])
    evaluate_model('K-Nearest Neighbors', model, X_test, y_test, results)


def MLPC(X_test, y_test, results, tuned_params):
    params = _pick_model_params(tuned_params, 'Multi-layer Perceptron', aliases=['MLP Classifier'])
    mlp_params = _strip_prefix(params, 'mlpc__')
    mlp = MLPClassifier(random_state=0)
    mlp.set_params(**_filter_params(mlp, mlp_params))
    evaluate_model('MLP Classifier', mlp, X_test, y_test, results)


def ELM(X_test, y_test, results, tuned_params):
    params = _pick_model_params(tuned_params, 'Extreme Learning Machine', aliases=['ELM Classifier'])
    elm_params = _strip_prefix(params, 'elm__')
    elm = ELMClassifier(random_state=69)
    elm.set_params(**_filter_params(elm, elm_params))
    evaluate_model('ELM Classifier', elm, X_test, y_test, results)

def plot_results(results):
    results_df = pd.DataFrame(results)

    metric_cols = ['Macro F1', 'Balanced Accuracy']
    metric_values = results_df[metric_cols].to_numpy().ravel()
    finite_vals = metric_values[np.isfinite(metric_values)]
    if finite_vals.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))
        if min_val == max_val:
            padding = 0.02
        else:
            padding = max(0.01, (max_val - min_val) * 0.2)
        y_min = max(0.0, min_val - padding)
        y_max = min(1.0, max_val + padding)

    x = np.arange(len(results_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, results_df['Macro F1'], width, label='F1')
    plt.bar(x + width / 2, results_df['Balanced Accuracy'], width, label='Balanced Accuracy')

    plt.xticks(x, results_df['Model'], rotation=20, ha='right')
    plt.ylim(y_min, y_max)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.tight_layout()

    output_path = _plots_dir() / 'model_performance_py.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f'Plot saved to: {output_path}')


def save_results_csv(results):
    results_df = pd.DataFrame(results)
    output_path = _model_results_path()
    results_df.to_csv(output_path, index=False)
    print(f'Results CSV saved to: {output_path}')


def __main__():
    data_dir = _data_dir()
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()
    tuned_params = _load_tuned_params()
    refit_models = _load_refit_models()
    X_train = None
    y_train = None

    results = []
    model_key_map = [
        ('Decision Tree', 'Decision Tree'),
        ('Random Forest', 'Random Forest'),
        ('Gradient Boosting', 'Gradient Boosting'),
        ('Support Vector Machine', 'Support Vector Machine'),
        ('K-Nearest Neighbors', 'K-Nearest Neighbors'),
        ('MLP Classifier', 'Multi-layer Perceptron'),
        ('ELM Classifier', 'Extreme Learning Machine'),
    ]

    for display_name, model_key in model_key_map:
        if model_key not in refit_models:
            if model_key == 'Extreme Learning Machine':
                print(
                    'Refit artifact missing Extreme Learning Machine. '
                    'Fitting ELM fallback model from tuned parameters.'
                )
                if X_train is None or y_train is None:
                    X_train, y_train = _load_train_data()
                elm_fallback_model = _build_elm_fallback(tuned_params)
                elm_fallback_model.fit(X_train, y_train)
                evaluate_model(display_name, elm_fallback_model, X_test, y_test, results)
                continue
            print(f'Skipping {display_name}: model not found in refit artifact ({model_key}).')
            continue
        evaluate_model(display_name, refit_models[model_key], X_test, y_test, results)

    save_results_csv(results)
    plot_results(results)


if __name__ == '__main__':
    __main__()