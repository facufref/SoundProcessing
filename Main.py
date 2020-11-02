from SoundDataManager import get_dataset_from_wavfile, get_train_test, pre_process, get_dataset_from_array
from SoundRecorder import *
from sklearn.metrics import classification_report
from SoundClassifier import *


def get_classifier():
    data, target, filenames = get_dataset_from_wavfile('wavfiles/Drones/', 'labels.csv')
    X_test, X_train, y_test, y_train, train_index, test_index = get_train_test(data, target)
    clf = SoundClassifier('randomForest')
    X_test, X_train = pre_process(X_test, X_train)
    clf.train_classifier(X_train, y_train)

    predictions = clf.get_predictions(X_test)
    print_predictions(predictions, filenames, test_index)
    print(f"Accuracy Train =  {str(clf.get_accuracy(X_train, y_train))}")
    print(f"Accuracy Test  =  {str(clf.get_accuracy(X_test, y_test))}")
    print(classification_report(y_test, predictions))
    # clf.show_feature_importance(data, target)
    # clf.print_decision_function(X_test)
    # clf.print_prediction_probability(X_test)
    return clf


def main():
    clf = get_classifier()


if __name__ == '__main__':
    main()
