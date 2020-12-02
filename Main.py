from sklearn.metrics import classification_report

from SoundClassifier import *
from SoundDataManager import get_dataset_from_wavfile, get_train_test, pre_process


def get_classifier():
    # feature_type: 'mfcc' or 'filter_banks'
    data, target, filenames = get_dataset_from_wavfile('wavfiles/Drones/', 'labels3.csv', 1.5, 'mfcc')
    X_test, X_train, y_test, y_train, train_index, test_index = get_train_test(data, target)
    X_test, X_train = pre_process(X_test, X_train)
    clf = SoundClassifier('svm')
    clf.train_classifier(X_train, y_train)

    predictions = clf.get_predictions(X_test)
    print_predictions(predictions, filenames, test_index)
    print(f"Accuracy Train =  {str(clf.get_accuracy(X_train, y_train))}")
    print(f"Accuracy Test  =  {str(clf.get_accuracy(X_test, y_test))}")
    print(classification_report(y_test, predictions))
    return clf


def main():
    clf = get_classifier()


if __name__ == '__main__':
    main()
