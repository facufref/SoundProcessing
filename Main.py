from SoundFileManager import get_dataset_from_wavfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def main():
    data, target = get_dataset_from_wavfile('wavfiles/', 'labels.csv')
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0, shuffle=False)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test, y_test))

if __name__ == '__main__':
    main()
