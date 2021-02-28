import numpy as np
from model import RandomForestWithFeatureSelection
from sklearn.model_selection import train_test_split

# paths convention;
X_PATH = './data/train_input/X.npy'
y_PATH = './data/train_input/y.npy'
MODEL_PATH = './data/model/model.joblib'

if __name__ == '__main__':

    # one can also collect the model hyperparams from the train docker run parameters
    # alternatively, one can also provide the hyperparams as an input config json and read it here
    model = RandomForestWithFeatureSelection(random_state=1, n_estimators=10, top_k=8)
    X = np.load(X_PATH)
    y = np.load(y_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    model.train(X_train, y_train)
    print("\nModel performance on the train dataset:")
    train_score = model.score(X_train, y_train)
    print("\nModel performance on the train dataset:")
    test_score = model.score(X_test, y_test)

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
