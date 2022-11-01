from data.DataController import DataController
from model.machine_learning.SklearnModelController import SklearnModelController
from model.train.SklearnModelTrainController import SklearnModelTrainController
from numpy import mean
from numpy import std
import warnings
warnings.filterwarnings('ignore')

class TestingModel:

    def sklearn_models_test(self):
        dataController = DataController()
        sklearnModelController = SklearnModelController()
        sklearnModelTrainController = SklearnModelTrainController()
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


test1 = TestingModel()
test1.sklearn_models_test()
