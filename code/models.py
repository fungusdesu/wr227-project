from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pathlib as path

def _data_dir():
    return path.Path(__file__).resolve().parent.parent / 'OULAD'

def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, min_samples_split=2)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Random Forest Accuracy: {accuracy:.4f}')

def decision_tree(X_train, y_train, X_test, y_test):    
    model = DecisionTreeClassifier(random_state=0, min_samples_split=2)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Decision Tree Accuracy: {accuracy:.4f}')

def gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Gradient Boosting Accuracy: {accuracy:.4f}')

def support_vector_machine(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(C=0.5, class_weight='balanced', random_state=0, max_iter=5000, dual='auto'))
    ])
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Support Vector Machine Accuracy: {accuracy:.4f}')

def gaussian_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB(var_smoothing=1e-12)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Gaussian Naive Bayes Accuracy: {accuracy:.4f}')

def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=31, weights='uniform', p=1))
    ])
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'K-Nearest Neighbors Accuracy: {accuracy:.4f}')

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