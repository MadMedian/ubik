from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd

if __name__ == '__main__':
    print('blabla')
    df = pd.DataFrame(np.random.choice(100, 1000))
    print(df.head())
    #
    #
    # dirpath = os.getcwd()
    # print("dirpath = ", dirpath, "\n")
    #
    # df.to_csv('/opt/src/output/output.csv')
    # sys.exit(0)

