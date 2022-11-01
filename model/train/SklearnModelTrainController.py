from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from numpy import absolute
from numpy import mean
from numpy import std
from amorf.metrics import average_relative_root_mean_squared_error


class SklearnModelTrainController:

    def fit_model(self, X, y, model, mse_type):
        if mse_type == "AVERAGE":
            return self.fit_model_average(X, y, model)
        elif mse_type == "DISTRIBUTED":
            return self.fit_model_distributed(X, y, model)
        else:
            print("Unsuported measure type")
            return 0

    def fit_model_average(self, X, y, model):
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

    def fit_model_distributed(self, X, y, model):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=33)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = []
        for i in range(y_test.shape[1]):
            mse.append(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))
        return mse
