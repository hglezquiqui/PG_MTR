from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from numpy import absolute
from numpy import mean
from numpy import std
from amorf.metrics import average_relative_root_mean_squared_error
from function.function import *


def fit_model(X, y, model, mse_type):
    if mse_type == "ARRMSE":
        return fit_model_arrmse(X, y, model)
    elif mse_type == "MSE":
        return fit_model_mse(X, y, model)
    elif mse_type == "RMSE":
        return fit_model_rmse(X, y, model)
    elif mse_type == "MAE":
        return fit_model_mae(X, y, model)
    elif mse_type == "MAPE":
        return fit_model_mape(X, y, model)
    else:
        print("Unsuported measure type")
        return 0


def fit_model_arrmse(X, y, model):
    results = []
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=25)
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = average_relative_root_mean_squared_error(y_pred, y_test)
        results.append(mse)
    return results


def fit_model_mse(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = []
    for i in range(y_test.shape[1]):
        mse.append(mean_squared_error(y_test[:, i], y_pred[:, i]))
    return mse


def fit_model_rmse(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = []
    for i in range(y_test.shape[1]):
        rmse.append(RMSE(y_test[:, i], y_pred[:, i]))
    return rmse


def fit_model_mae(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = []
    for i in range(y_test.shape[1]):
        mae.append(MAE(y_test[:, i], y_pred[:, i]))
    return mae


def fit_model_mape(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mape = []
    for i in range(y_test.shape[1]):
        mape.append(MAPE(y_test[:, i], y_pred[:, i]))
    return mape