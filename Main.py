from SoundDataManager import get_dataset_from_wavfile, get_train_test, pre_process
from sklearn.metrics import classification_report
from classifiers.SoundClassifier import *


def main():
    data, target, filenames = get_dataset_from_wavfile('wavfiles/Violins/', 'labels.csv')
    X_test, X_train, idx1, idx2, y_test, y_train = get_train_test(data, filenames, target)
    clf = SoundClassifier('svm')
    X_test, X_train = pre_process(X_test, X_train)
    clf.train_classifier(X_train, y_train)

    predictions = clf.get_predictions(X_test)
    print_predictions(predictions, filenames, idx2)
    print(f"Accuracy Train =  {str(clf.get_accuracy(X_train, y_train))}")
    print(f"Accuracy Test  =  {str(clf.get_accuracy(X_test, y_test))}")
    print(classification_report(y_test, predictions))
    # clf.show_feature_importance(data, target)
    # clf.print_decision_function(X_test)
    # clf.print_prediction_probability(X_test)


if __name__ == '__main__':
    main()
