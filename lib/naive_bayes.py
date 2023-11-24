import numpy as np
import math

class NaiveBayesClassifier():
    def __init__(self, alpha: int) -> None:
        self.numerical_columns = []
        self.categorical_columns = []
        self.y_train = []
        self.target_values = []
        self.alpha = alpha
        self.target_probs = {}
        self.categorical_conditional_probs = []

    def fit(self, numerical_columns, categorical_columns, y_train):
        self.y_train = y_train
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.CategoricalProbability()

    def predictProbaNumerical(self, x_test_numerical: list[list[float]]):
        res = []
        for x_test_row in x_test_numerical:
            numerical_probabilities = self.GaussianProbability(x_test_row)
            res.append(numerical_probabilities)
        return res

    def GaussianProbability(self, x_numerical_test: list[float]):
        y_train_np = np.array(self.y_train)
        target_values = np.unique(y_train_np)
        # log_numerical_probabilities = []
        target_probabilities = []
        for target_value in target_values:
            # log_target_probabilities = []
            target_given_all_probabilities = self.target_probs[target_value]
            for col, x_train_values in enumerate(self.numerical_columns):
                indices = np.where(y_train_np == target_value)
                x_train_values = np.array(x_train_values)[indices]

                var = np.var(x_train_values)
                avg = np.average(x_train_values)
                xi = x_numerical_test[col]

                # log_target_given_x_probability = (-0.5 * np.log(2 * np.pi * var)) - (0.5 * (((xi - avg) ** 2) / var))
                x_given_target_probability = (1 / math.sqrt(2 * np.pi * var)) * math.exp(- ((xi - avg) ** 2) / (2 * var))
                # log_target_probabilities.append(log_target_given_x_probability)
                target_given_all_probabilities *= x_given_target_probability
            # log_numerical_probabilities.append(log_target_probabilities)
            target_probabilities.append(target_given_all_probabilities)
        
        prob_sum = np.sum(target_probabilities)
        target_probabilities = [target_probability / prob_sum for target_probability in target_probabilities]

        return target_probabilities

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

