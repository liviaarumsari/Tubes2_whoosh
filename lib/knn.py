from collections import defaultdict
import math


def calculate_manhattan_distance(point1, point2):
    """
    Function to calculate manhattan distance between point 1 and point 2 of dim len(point1)
    :param point1:
    :param point2:
    :return: manhattan distance between point1 and point2
    """
    dist = 0
    for i in range(len(point1)):
        dist += abs(point1[i] - point2[i])
    return dist


def calculate_euclidean_distance(point1, point2):
    """
     Function to calculate euclidean distance between point 1 and point 2 of dim len(point1)
    :param point1:
    :param point2:
    :return: euclidean distance between point1 and point2
    """
    dist = 0
    for i in range(len(point1)):
        dist += (point1[i] - point2[i]) ** 2
    dist = math.sqrt(dist)

    return dist


class KNeighborsClassifier():
    def __init__(self, n_neighbors, metric, weight='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = []
        self.y_train = []
        self.weight = weight

    def fit(self, X_train, y_train):
        """
        Save the training data to the class
        :param X_train:
        :param y_train:
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict the y values for each X in X_test
        :param X_test:
        :return prediction for each X in X_test:
        """
        nearest_neighbors = self.__get_nearest_neighbors(X_test)  # get the nearest neighbors in Xs for each X_test
        prediction = []
        for neighbors in nearest_neighbors:
            prediction_map = defaultdict(int)
            for neighbor_value, neighbor_key in neighbors:
                if self.weight == 'uniform':
                    prediction_map[neighbor_key] += neighbor_value
                elif self.weight == 'distance':
                    prediction_map[neighbor_key] += (1 if neighbor_value == 0 else 1 / neighbor_value)
            max_key = max(prediction_map, key=prediction_map.get)
            prediction.append(max_key)
        return prediction

    def predict_proba(self, X_test):
        """
        Function to calculate the probability of all categories in the target column for each X in X_test
        :param X_test:
        :return:
        """
        nearest_neighbors = self.__get_nearest_neighbors(X_test)  # get the nearest neighbors in Xs for each X_test
        prediction = []
        for neighbors in nearest_neighbors:
            prediction_map = defaultdict(int)
            for neighbor_value, neighbor_key in neighbors:
                if self.weight == 'uniform':
                    prediction_map[neighbor_key] += neighbor_value
                elif self.weight == 'distance':
                    prediction_map[neighbor_key] += (1 if neighbor_value == 0 else 1 / neighbor_value)
            prediction_list = [0 for i in range(len(self.__get_possible_targets()))]
            sum_values = sum(prediction_map.values())
            for key in prediction_map.keys():
                prediction_list[key] = prediction_map[key] / sum_values
            prediction.append(prediction_list)
        return prediction

    def __get_possible_targets(self):
        return set(self.y_train)

    def __get_nearest_neighbors(self, X_test):
        """
        Find the nearest neighbors for each X in X_test
        :param X_test:
        :return The nearest neighbors with format list of (dist value, y_train) for each X in X_test:
        """
        dist = self.__calculate_test_to_train_distance(X_test)
        sorted_dist = [sorted(sublist, key=lambda x: x[0])[:self.n_neighbors] for sublist in dist]
        return sorted_dist

    def __calculate_test_to_train_distance(self, X_test):
        """
        Find the distance for each X in X_test to each points in X_train
        :param X_test:
        :return the list of tuple (distance, y_train):
        """
        X_train_length = len(self.X_train)
        X_test_length = len(X_test)
        dist = [[0 for _ in range(X_train_length)] for _ in range(X_test_length)]
        for i in range(X_test_length):
            for j in range(X_train_length):
                if self.metric == 'manhattan':
                    dist[i][j] = (calculate_manhattan_distance(X_test[i], self.X_train[j]), self.y_train[j])
                elif self.metric == 'euclidean':
                    dist[i][j] = (calculate_euclidean_distance(X_test[i], self.X_train[j]), self.y_train[j])
        return dist

    def score(self, X_test, y_test):
        """
        Predict and calculate the accuracy score
        :param X_test:
        :param y_test:
        :return the accuracy of the y calculated:
        """
        prediction = self.predict(X_test)
        correct_cnt = sum(pred == actual for pred, actual in zip(prediction, y_test))
        return correct_cnt / len(y_test)


# main to test with dummy data
if __name__ == "__main__":
    KNN_manhattan = KNeighborsClassifier(1, "manhattan")
    KNN_manhattan.fit([[2, 3], [3, 4], [1, 2], [6, 7], [9, 0]], [2, 2, 3, 3, 4])
    print(KNN_manhattan.predict([[2, 3], [3, 4], [1, 2], [6, 7], [9, 0]]))
