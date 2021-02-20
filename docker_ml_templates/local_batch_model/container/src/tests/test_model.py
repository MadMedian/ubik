import unittest
from ..model import RandomForestWithFeatureSelection
from sklearn.model_selection import train_test_split
import os
import numpy as np


def create_dataset(n_rows=1000, n_feats=10, pos_loc=2.0, neg_loc=0.0,
                   pos_scale=3.0, neg_scale=3.0):
    X_pos = np.random.normal(pos_loc, pos_scale, size=(n_rows, n_feats))
    X_neg = np.random.normal(neg_loc, neg_scale, size=(n_rows, n_feats))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_rows), np.zeros(n_rows)])
    return X, y


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.random_state = 1
        self.filepath = './src/tests/model.joblib'

    def test_model(self):
        unit = RandomForestWithFeatureSelection(random_state=self.random_state, n_estimators=10, top_k=8)
        X, y = create_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=self.random_state)

        model = unit.train(X_train, y_train)
        train_score = unit.score(X_train, y_train)
        test_score = unit.score(X_test, y_test)
        self.assertGreater(train_score['precision_score'], 0.95)
        self.assertGreater(test_score['precision_score'], 0.75)
        for score_train, score_test in zip(train_score.values(), test_score.values()):
            self.assertGreater(score_train, score_test)

    def test_save_load(self):
        model = RandomForestWithFeatureSelection(random_state=self.random_state, n_estimators=5, top_k=6)
        X, y = create_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=self.random_state)
        model = model.train(X_train, y_train)

        model.save(self.filepath)
        unit = RandomForestWithFeatureSelection.load(self.filepath)

        train_score = unit.score(X_train, y_train)
        test_score = unit.score(X_test, y_test)
        self.assertGreater(train_score['precision_score'], 0.9)
        self.assertGreater(test_score['precision_score'], 0.7)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
            return


if __name__ == '__main__':
    unittest.main()
