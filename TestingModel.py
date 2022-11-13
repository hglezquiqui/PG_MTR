from data.DataController import *
from model.machine_learning.SklearnModelController import *
from model.train.SklearnModelTrainController import *
from function.function import *
from function.graphic import *
from numpy import mean
from numpy import std
import warnings
warnings.filterwarnings('ignore')


def MTR_EA(mse_type):
    X, y = create_random_regression_problem(500, 10, 5, 5, 1, 0.5)
    model_type = ['LR', 'KNN', 'DT', 'RF', 'MRF', 'SVR', 'RC_SVR', 'GB']
    result = {}
    result.fromkeys(model_type)
    for m in model_type:
        model = build_model_by_type(m)
        result[m] = fit_model(X, y, model, mse_type)
        print('%s_%s: %.3f (%.3f)' % (mse_type, m, mean(result[m]), std(result[m])))
    print(result)


def GMLR1(dataset):
    X_train, X_test, y_train, y_test = read_mtr_arff(dataset)
    xk, zk, cri, fobj = lasso_pgfit(X_train, y_train)
    plot_and_reconstructed(dataset, xk, zk)
    plot_objective_function_values(dataset, cri, fobj)

