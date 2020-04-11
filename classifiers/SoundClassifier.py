from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class SoundClassifier:
    def __init__(self, algorithm):
        if algorithm == 'knn':
            self._classifier = KNeighborsClassifier(n_neighbors=6)
        elif algorithm == 'linear':
            self._classifier = LogisticRegression()
        elif algorithm == 'linearMulti':
            self._classifier = LinearSVC()
        else:
            print('Algorithm not found')

    def train_classifier(self, X_train, y_train):
        self._classifier.fit(X_train, y_train)

    def print_predictions(self, X_test, filenames, idx2):
        for i in range(0, len(X_test)):
            prediction = self._classifier.predict(X_test[i].reshape(1, -1))
            print(f" The file '{filenames[idx2[i]]} is {str(prediction)}")

    def print_accuracy(self, X_test, y_test):
        accuracy = self._classifier.score(X_test, y_test)
        print(f"Accuracy =  {str(accuracy)}")
