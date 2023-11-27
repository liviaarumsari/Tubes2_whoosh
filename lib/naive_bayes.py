import numpy as np
import pandas as pd
import math


class NaiveBayesGaussian():
    def __init__(self) -> None:
        self.numerical_columns = []
        self.y_train = []
        self.target_probs = {}
        self.target_values = []

    def fit(self, numerical_columns: pd.DataFrame, y_train) -> None:
        """
        Save numerical target training data into class, count target probabilities
        :param numerical_columns:
        :param y_train:
        :return:
        """
        self.y_train = y_train.values
        self.numerical_columns = numerical_columns.values.T

        total_count = len(self.y_train)

        for value in np.unique(self.y_train):
            count_value = np.sum(self.y_train == value)
            self.target_probs[value] = count_value / (total_count + len(np.unique(self.y_train)))

    def predict_proba(self, x_test_numerical: np.ndarray) -> list:
        """
        Return list of normalized probability of x_test_numerical to target classification
        :param x_test_numerical:
        :return:
        """
        res = []
        for x_test_row in x_test_numerical:
            numerical_probabilities = self.gaussian_proba(x_test_row)
            res.append(numerical_probabilities)
        return res

    def gaussian_proba(self, x_numerical_test):
        """
        Return list of gaussian probability of x_numerical_test
        :param x_numerical_test:
        :return:
        """
        y_train_np = np.array(self.y_train)
        target_values = np.unique(y_train_np)
        target_probabilities = []
        for target_value in target_values:
            log_target_given_all_probabilities = math.log(self.target_probs[target_value])
            for col, x_train_values in enumerate(self.numerical_columns):
                indices = np.where(y_train_np == target_value)
                indices = [index for index, value in enumerate(self.y_train) if value == target_value]
                x_train_values = np.array(x_train_values)[indices]

                var = np.var(x_train_values)
                avg = np.average(x_train_values)
                xi = x_numerical_test[col]

                log_target_given_x_probability = (-0.5 * math.log(2 * np.pi * var)) - (0.5 * (((xi - avg) ** 2) / var))
                log_target_given_all_probabilities += log_target_given_x_probability
            target_probabilities.append(math.exp(log_target_given_all_probabilities))

        prob_sum = np.sum(target_probabilities)
        target_probabilities = [target_probability / prob_sum for target_probability in target_probabilities]

        return target_probabilities


class NaiveBayesCategorical():
    def __init__(self) -> None:
        self.categorical_columns = []
        self.y_train = []
        self.target_probs = {}
        self.conditional_probs = []

    def fit(self, categorical_columns: pd.DataFrame, y_train) -> None:
        """
        Save categorical target training data into class, count target and conditional probabilities
        :param categorical_columns:
        :param y_train:
        :return:
        """
        self.y_train = y_train.values
        self.categorical_columns = categorical_columns.values.T

        total_count = len(self.y_train)

        for value in np.unique(self.y_train):
            count_value = np.sum(self.y_train == value)
            self.target_probs[value] = count_value / (total_count + len(np.unique(self.y_train)))

        for column in self.categorical_columns:
            cond_probs = dict()
            for value1 in np.unique(column):
                for value2 in np.unique(self.y_train):
                    count_value1_value2 = np.sum((column == value1) & (self.y_train == value2))
                    count_value2 = np.sum(self.y_train == value2)
                    cond_probs[f'{value1}-{value2}'] = (count_value1_value2 + 1) / (count_value2 + (1 * len(np.unique(column))))
            self.conditional_probs.append(cond_probs)

    def predict_proba(self, x_test: np.ndarray) -> list:
        """
        Return list of normalized probability of x_test to target classification
        :param x_test:
        :return:
        """
        predicts = []
        for row in x_test:
            row_probabilities = []
            # Iterate sorted unique value of target column
            for target in np.unique(self.y_train):
                prior_prob = self.target_probs[target]
                for i in range(len(x_test[0])):
                    cond_prob_key = f'{row[i]}-{target}'
                    cond_prob = self.conditional_probs[i][cond_prob_key]

                    prior_prob *= cond_prob

                row_probabilities.append(prior_prob)

            sum_prob = np.sum(row_probabilities)
            normalized_prob = row_probabilities / sum_prob
            predicts.append(normalized_prob.tolist())

        return predicts


# Function test with dummy data
if __name__ == "__main__":
    nv = NaiveBayesCategorical()
    nv.fit(pd.DataFrame([[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0]]), pd.DataFrame([1, 1, 0, 0]))
    print(nv.predict_proba(np.array([[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0]])))
