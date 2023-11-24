import numpy as np

class NaiveBayesClassifier():
    def __init__(self, alpha: int) -> None:
        self.numerical_columns = []
        self.categorical_columns = []
        self.y_train = []
        self.alpha = alpha
        self.target_probs = {}
        self.categorical_conditional_probs = []

    def fit(self, numerical_columns, categorical_columns, y_train):
        self.y_train = y_train
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    def predict(self, x_test):
        pass

    def GaussianProbability(self, x_test):
        pass

    def CategoricalProbability(self):
        total_count = len(self.y_train)

        for value in np.unique(self.y_train):
            count_value = np.sum(self.y_train == value)
            self.target_probs[value] = (count_value + self.alpha) / (total_count + (len(np.unique(self.y_train)) * self.alpha))

        for column in self.categorical_columns:
            cond_probs = dict()
            for value1 in np.unique(column):
                for value2 in np.unique(self.y_train):
                    count_value1_value2 = np.sum((column == value1) & (self.y_train == value2))
                    count_value2 = np.sum(self.y_train == value2)
                    cond_probs[f'{value1}-{value2}'] = (count_value1_value2 + self.alpha) / (count_value2 + (self.alpha * len(np.unique(column))))
            self.categorical_conditional_probs.append(cond_probs)



if __name__ == "__main__":
    nv = NaiveBayesClassifier(1)
    nv.fit([], [[0,0,0,1],[1,1,1,0],[1,1,0,0]], [1,1,0,0])
    nv.CategoricalProbability()

