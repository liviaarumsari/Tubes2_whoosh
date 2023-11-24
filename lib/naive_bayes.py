class NaiveBayesClassifier():
    def __init__(self, alpha):
        self.numerical_columns = []
        self.categorical_columns = []
        self.x_train = []
        self.y_train = []
        self.target_values = []
        self.alpha = alpha

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass

    def GaussianProbability(self, numerical_train: list[list[float]], y_train: list[int], x_numerical_test: list[float]):
        y_train = np.array(y_train)
        target_values = np.unique(y_train)
        log_numerical_probabilities = []
        for target_value in target_values:
            log_target_probabilities = []
            for col, x_train_values in enumerate(numerical_train):
                indices = np.where(y_train == target_value)
                x_train_values = np.array(x_train_values)[indices]

                var = np.var(x_train_values)
                avg = np.average(x_train_values)
                xi = x_numerical_test[col]

                log_target_given_x_probability = (-0.5 * np.log(2 * np.pi * var)) - (0.5 * (((xi - avg) ** 2) / var))
                log_target_probabilities.append(log_target_given_x_probability)
            log_numerical_probabilities.append(log_target_probabilities)
        
        return log_numerical_probabilities

    def CategoricalProbability(self, x_test):
        pass


if __name__ == "__main__":
    pass
