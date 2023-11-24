class NaiveBayesClassifier():
    def __init__(self, alpha, ):
        self.numerical_columns = []
        self.categorical_columns = []
        self.x_train = []
        self.y_train = []
        self.alpha = alpha

    def fit(self, x_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def GaussianProbability(self, x_test, y_test):
        pass

    def CategoricalProbability(self, x_test, y_test):
        pass


if __name__ == "__main__":
    pass
