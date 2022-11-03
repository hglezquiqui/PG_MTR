from sklearn.datasets import make_regression
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import os


def create_random_regression_problem(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_targets=5,
    random_state=1,
    noise=0.5
):
    return make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        random_state=random_state,
        noise=noise
    )


def read_mtr_arff(dataset):
    sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]),"PG_MTR/data/repositories"))
    dataset_name = dataset['name'][0]
    q = dataset['output'][0]
    # print(q)
    pthTr = Path(sys.path[0], dataset_name+'/' + dataset_name+'-train.arff')
    print(pthTr)
    xTr = arff.loadarff(pthTr)
    df = pd.DataFrame(xTr[0])

    XTr = df.iloc[:, :-q]
    YTr = df.iloc[:, -q:]
    pthTe = Path(sys.path[0], dataset_name+'/' + dataset_name+'-test.arff')
    xTe = arff.loadarff(pthTe)
    df = pd.DataFrame(xTe[0])

    XTe = df.iloc[:, :-q]
    YTe = df.iloc[:, -q:]
    # print(YTr)
    sc = StandardScaler()
    XTr = sc.fit_transform(XTr)
    XTe = sc.transform(XTe)
    YTr = sc.fit_transform(YTr)
    YTe = sc.transform(YTe)

    return XTr, XTe, YTr, YTe
