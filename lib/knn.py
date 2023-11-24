from collections import Counter


def calculate_manhattan_distance(point1, point2):
    dist = 0
    for i in range(len(point1)):
        dist += abs(point1[i] - point2[i])
    return dist


class KNeighborsClassifier():
    def __init__(self, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        """
        Save the training data to the class
        :param X_train:
        :param y_train:
        :return:
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        nearest_neighbors = self.get_nearest_neighbors(X_test)  # get the nearest neighbors in Xs for each X_test
        nearest_neighbors_y = self.map_nearest_neighbors_to_target(nearest_neighbors)
        prediction = []

        for neighbors in nearest_neighbors_y:
            neighbor_counts = Counter(neighbors)
            most_common_neighbor, count = neighbor_counts.most_common(1)[0]
            prediction.append(most_common_neighbor)
        return prediction

    def map_nearest_neighbors_to_target(self, nearest_neighbors):
        nearest_neighbors_y = []
        for neighbours in nearest_neighbors:
            curr_y = []
            for neighbour in neighbours:
                curr_y.append(self.y_train[neighbour])
            nearest_neighbors_y.append(curr_y)
        return nearest_neighbors_y

    def get_nearest_neighbors(self, X_test):
        dist = self.calculate_test_to_train_distance(X_test)
        sorted_dist_with_indices = [
            sorted(enumerate(sublist), key=lambda x: x[1], reverse=False) for sublist in dist
        ]
        nearest_neighbors_with_indices = [
            sublist[:self.n_neighbors] for sublist in sorted_dist_with_indices
        ]

        # Keeping only the original indices
        nearest_neighbors = [
            [index for index, _ in sublist] for sublist in nearest_neighbors_with_indices
        ]
        return nearest_neighbors

    def calculate_test_to_train_distance(self, X_test):
        X_train_length = len(self.X_train)
        X_test_length = len(X_test)
        dist = [[0 for _ in range(X_train_length)] for _ in range(X_test_length)]
        for i in range(X_test_length):
            for j in range(X_train_length):
                if self.metric == 'manhattan':
                    dist[i][j] = calculate_manhattan_distance(X_test[i], self.X_train[j])
        return dist

    def score(self, X_test, y_test):
        prediction = self.predict(X_test)
        correct_cnt = sum(pred == actual for pred, actual in zip(prediction, y_test))
        return correct_cnt / len(y_test)


if __name__ == "__main__":
    KNN_manhattan = KNeighborsClassifier(1, "manhattan")
    KNN_manhattan.fit([[2, 3], [3, 4], [1, 2], [6, 7], [9, 0]], [2, 2, 3, 3, 4])
    KNN_manhattan.predict([[2, 3], [3, 4], [1, 2], [6, 7], [9, 0]])
