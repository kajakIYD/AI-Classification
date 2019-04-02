from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def split_dataset(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def fit_SGD(X, y, loss="log", penalty="l2", max_iter=500, random_state=42):
    clf = SGDClassifier(loss=loss, penalty=penalty, max_iter=max_iter, random_state=random_state)
    clf.fit(X, y)
    return clf


def fit_SVC(X, y, kernel="linear", C=0.01, probability=True):
    clf = svm.SVC(kernel=kernel, C=C, probability=probability)
    clf.fit(X, y)
    return clf


def fit_RandomForrest(X, y, n_estimators=100, max_depth=2, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=random_state)
    clf.fit(X, y)
    return clf


def fit_DecisionTree(X, y, random_state=42):
    clf = DecisionTreeClassifier(random_state=random_state)
    clf = clf.fit(X, y)
    return clf
