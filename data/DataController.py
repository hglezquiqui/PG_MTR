from sklearn.datasets import make_regression


class DataController:

    def create_random_regression_problem(
        self,
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
