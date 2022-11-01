from data import DataController as dataController
from model.machine_learning import SklearnModelController as sklearnModelController
from model.train import SklearnModelTrainController as sklearnModelTrainController
from numpy import mean
from numpy import std
import seaborn as sbn
import warnings
warnings.filterwarnings('ignore')


def sklearn_models_test():
    X, y = dataController.create_random_regression_problem(
        500, 10, 5, 5, 1, 0.5
    )
    model_type = ['LR', 'KNN', 'DT', 'RF', 'MRF', 'SVR', 'RC_SVR', 'GB']
    result = {}
    result.fromkeys(model_type)
    for m in model_type:
        model = sklearnModelController.build_model_by_type(m)
        result[m] = sklearnModelTrainController.fit_model(
            X, y, model, "AVERAGE")
        print('MSE_%s: %.3f (%.3f)' %
              (m, mean(result[m]), std(result[m])))
    print(result)


sklearn_models_test()
