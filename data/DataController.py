from sklearn.datasets import make_regression

def create_random_regression_problem(
    n_samples, 
    n_features, 
    n_informative, 
    n_targets, 
    random_state, 
    noise
    ):
    return make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        n_targets=n_targets, 
        random_state=random_state, 
        noise=noise
        )

