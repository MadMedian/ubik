import numpy as np
from model import RandomForestWithFeatureSelection

# paths convention
X_BATCH_PATH = './data/pred_input/X_batch.npy'
MODEL_PATH = './data/model/model.joblib'
y_PRED_PATH = './data/pred_output/y_pred_batch.npy'

if __name__ == '__main__':

    model = RandomForestWithFeatureSelection.load(MODEL_PATH)
    X_batch = np.load(X_BATCH_PATH)
    y_pred_batch = model.predict(X_batch)

    np.save(y_PRED_PATH, y_pred_batch)
    print(f"\nPredictions saved to {y_PRED_PATH}")
