from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError
import joblib


class RandomForestWithFeatureSelection(object):
    """
    We create a model made of a feature selector and a random forest classifier
    """
    def __init__(self, top_k=10, n_estimators=50, n_jobs=-1, random_state=15):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.selector = None
        self.classifier = None
        self.n_jobs = n_jobs
        self.random_state = random_state

    def __fit_selector__(self, X, y):
        sel = SelectKBest(score_func=mutual_info_classif, k=self.top_k)
        sel.fit(X, y)
        self.selector = sel

    def __transform_selector__(self, X):
        return self.selector.transform(X)

    def __fit_classifier__(self, X, y):
        rf = RandomForestClassifier(n_estimators=self.n_estimators,
                                    random_state=self.random_state,
                                    n_jobs=self.n_jobs)
        rf.fit(X, y)
        self.classifier = rf

    def __transform_classifier__(self, X):
        return self.classifier.predict(X)

    def train(self, X, y):
        """trains whole model"""
        self.__fit_selector__(X, y)
        X_sel = self.__transform_selector__(X)
        self.__fit_classifier__(X_sel, y)
        return self

    def predict(self, X):
        """entry point for prediction"""
        if self.selector is None or self.classifier is None:
            print("Exception: Train the model first!")
            raise NotFittedError
        X_sel = self.__transform_selector__(X)
        y_pred = self.__transform_classifier__(X_sel)
        return y_pred

    def score(self, X, y):
        """evaluate trained model"""
        y_pred = self.predict(X)
        eval_report = {}
        for metric in [accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score]:
            key = metric.__name__
            value = metric(y, y_pred)
            eval_report[key] = value
            print(f"{key[:-6]:>17} : {value:.3f}")
        return eval_report

    def save(self, filepath):
        joblib.dump(self, filepath)
        return

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
