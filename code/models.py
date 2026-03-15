from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
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
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Gradient Boosting Accuracy: {accuracy:.4f}')

def support_vector_machine(X_train, y_train, X_test, y_test):
    model = svm.SVC(kernel='linear', random_state=0)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Support Vector Machine Accuracy: {accuracy:.4f}')

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

if __name__ == '__main__':
    __main__()