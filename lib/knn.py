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
        nearest_neighbors = self.get_nearest_neighbors(X_test)  # get the nearest neighbors in Xs for each X_test
        prediction = []
        print(self.weight)
        for neighbors in nearest_neighbors:
            prediction_map = defaultdict(int)
            for neighbor_value, neighbor_key in neighbors:
                if self.weight == 'uniform':
                    prediction_map[neighbor_key] += neighbor_value
                elif self.weight == 'distance':
                    prediction_map[neighbor_key] += (1 if neighbor_value == 0 else 1/neighbor_value)
            max_key = max(prediction_map, key=prediction_map.get)
            prediction.append(max_key)
        return prediction

    def map_nearest_neighbors_to_target(self, nearest_neighbors):
        """
        Get the target for each nearest neighbors(List of X indexes of nearest neighbors)
        :param nearest_neighbors:
        :return:
        """
        nearest_neighbors_y = []
        for neighbours in nearest_neighbors:
            curr_y = []
            for neighbour in neighbours:
                curr_y.append(self.y_train[neighbour])
            nearest_neighbors_y.append(curr_y)
        return nearest_neighbors_y

    def get_nearest_neighbors(self, X_test):
        """
        Find the nearest neighbors for each X in X_test
        :param X_test:
        :return The index of nearest neighbors for each X in X_test:
        """
        dist = self.calculate_test_to_train_distance(X_test)
        sorted_dist = [sorted(sublist, key=lambda x: x[0])[:self.n_neighbors] for sublist in dist]
        return sorted_dist

    def calculate_test_to_train_distance(self, X_test):
        """
        Find the distance for each X in X_test to each points in X_train
        :param X_test:
        :return:
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
