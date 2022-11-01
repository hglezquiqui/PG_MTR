from data.DataController import DataController
from model.machine_learning.SklearnModelController import SklearnModelController
from model.train.SklearnModelTrainController import SklearnModelTrainController


class TestingModel:

    def sklearn_models_test(self):
        X, y = DataController.create_random_regression_problem(
            500, 10, 5, 5, 1, 0.5
        )
        model_type = ['LR', 'KNN', 'DT', 'RF', 'MRF', 'SVR', 'RC_SVR', 'GB']
        result = {}
        result.fromkeys(model_type)
        for m in model_type:
            model = SklearnModelController.build_model_by_type(model_type)
            result[m] = SklearnModelTrainController.fit_model(X, y, model, "DISTRIBUTED")
        print(result)
