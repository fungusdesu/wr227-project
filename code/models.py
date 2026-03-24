from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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


def _plots_dir():
    plots_dir = path.Path(__file__).resolve().parent / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'{model_name} Accuracy: {accuracy:.4f}')
    print(f'{model_name} Balanced Accuracy: {balanced_accuracy:.4f}')
    print(f'{model_name} F1 Score: {f1:.4f}')
    print()

    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy,
        'F1 Score': f1
    })


def random_forest(X_train, y_train, X_test, y_test, results):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=50,
        random_state=0,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=False
    )
    evaluate_model('Random Forest', model, X_train, y_train, X_test, y_test, results)

def gradient_boosting(X_train, y_train, X_test, y_test, results):
    model = GradientBoostingClassifier(
        n_estimators=100,
        subsample =1,
        learning_rate = 0.2,
        min_samples_split=2,
        min_samples_leaf=4,
        max_depth=5,
        max_features='auto',
        )
    evaluate_model('Gradient Boosting', model, X_train, y_train, X_test, y_test, results)


def support_vector_machine(X_train, y_train, X_test, y_test, results):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(
            C=0.785,
            class_weight='balanced',
            random_state=0,
            max_iter=5000,
            dual='auto'
        ))
    ])
    evaluate_model('Support Vector Machine', model, X_train, y_train, X_test, y_test, results)


def k_nearest_neighbors(X_train, y_train, X_test, y_test, results):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(
            n_neighbors=7,
            weights='uniform',
            p=1,
            metric='minkowski',
            leaf_size=50
        ))
    ])
    evaluate_model('K-Nearest Neighbors', model, X_train, y_train, X_test, y_test, results)

def MLPC(X_train, y_train, X_test, y_test, results):
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=5000,
        random_state=0
    )
    evaluate_model('MLP Classifier', mlp, X_train, y_train, X_test, y_test, results)

def ELM(X_train, y_train, X_test, y_test, results):
    elm = ELMClassifier(
        n_neurons=500,
        ufunc='tanh',
        alpha=1e-4,
        random_state=69
    )
    evaluate_model('ELM Classifier', elm, X_train, y_train, X_test, y_test, results)

def plot_results(results):
    results_df = pd.DataFrame(results)

    x = np.arange(len(results_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, results_df['Accuracy'], width, label='Accuracy')
    plt.bar(x, results_df['Balanced Accuracy'], width, label='Balanced Accuracy')
    plt.bar(x + width, results_df['F1 Score'], width, label='F1 Score')

    plt.xticks(x, results_df['Model'], rotation=20, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.tight_layout()

    output_path = _plots_dir() / 'model_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f'Plot saved to: {output_path}')


def __main__():
    data_dir = _data_dir()
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()

    results = []

    random_forest(X_train, y_train, X_test, y_test, results)
    gradient_boosting(X_train, y_train, X_test, y_test, results)
    support_vector_machine(X_train, y_train, X_test, y_test, results)
    k_nearest_neighbors(X_train, y_train, X_test, y_test, results)
    MLPC(X_train, y_train, X_test, y_test, results)
    ELM(X_train, y_train, X_test, y_test, results)
    plot_results(results)


if __name__ == '__main__':
    __main__()