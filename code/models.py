from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import pandas as pd
import pathlib as path

def _data_dir():
    return path.Path(__file__).resolve().parent.parent / 'OULAD'


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'{model_name} Accuracy: {accuracy:.4f}')
    print(f'{model_name} Balanced Accuracy: {balanced_accuracy:.4f}')
    print(f'{model_name} F1 Score: {f1:.4f}')

def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, min_samples_split=2)
    evaluate_model('Random Forest', model, X_train, y_train, X_test, y_test)

def decision_tree(X_train, y_train, X_test, y_test):    
    model = DecisionTreeClassifier(random_state=0, min_samples_split=2)
    evaluate_model('Decision Tree', model, X_train, y_train, X_test, y_test)

def gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(n_estimators=100)
    evaluate_model('Gradient Boosting', model, X_train, y_train, X_test, y_test)

def support_vector_machine(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(C=0.5, class_weight='balanced', random_state=0, max_iter=5000, dual='auto'))
    ])
    evaluate_model('Support Vector Machine', model, X_train, y_train, X_test, y_test)

def gaussian_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB(var_smoothing=1e-12)
    evaluate_model('Gaussian Naive Bayes', model, X_train, y_train, X_test, y_test)

def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=31, weights='uniform', p=1))
    ])
    evaluate_model('K-Nearest Neighbors', model, X_train, y_train, X_test, y_test)

def __main__():
    data_dir = _data_dir()
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv').squeeze()
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv').squeeze()


    random_forest(X_train, y_train, X_test, y_test)
    decision_tree(X_train, y_train, X_test, y_test)
    gradient_boosting(X_train, y_train, X_test, y_test)
    support_vector_machine(X_train, y_train, X_test, y_test)
    gaussian_naive_bayes(X_train, y_train, X_test, y_test)
    k_nearest_neighbors(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    __main__()