from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor


class SklearnModelController:

    def build_model_by_type(model_type):
        if model_type == 'LR':
            return LinearRegression()
        elif model_type == 'KNN':
            return KNeighborsRegressor(algorithm='auto',
                                       leaf_size=30,
                                       metric='minkowski',
                                       metric_params=None,
                                       n_jobs=1,
                                       n_neighbors=10,  # default is 5
                                       p=2,         # p=2 is equivalent to euclidian distance
                                       weights='uniform')
        elif model_type == 'DT':
            return DecisionTreeRegressor()
        elif model_type == 'RF':
            return RandomForestRegressor(max_depth=2, random_state=0)
        elif model_type == 'MRF':
            return MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0))
        elif model_type == 'SVR':
            return MultiOutputRegressor(LinearSVR())
        elif model_type == 'RC_SVR':
            return RegressorChain(LinearSVR())
        elif model_type == 'GB':
            return MultiOutputRegressor(GradientBoostingRegressor())
        print('Unsuported classifier')
        return 0


