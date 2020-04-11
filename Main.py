from SoundDataManager import get_dataset_from_wavfile, get_train_test
from classifiers.SoundClassifier import SoundClassifier


def main():
    data, target, filenames = get_dataset_from_wavfile('wavfiles/Drones/', 'labels.csv')
    X_test, X_train, idx1, idx2, y_test, y_train = get_train_test(data, filenames, target)
    clf = SoundClassifier('linear')
    clf.train_classifier(X_train, y_train)
    clf.print_predictions(X_test, filenames, idx2)
    clf.print_accuracy(X_test, y_test)


if __name__ == '__main__':
    main()
