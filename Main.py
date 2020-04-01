from SoundFileManager import get_dataset_from_wavfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    data, target, filenames = get_dataset_from_wavfile('wavfiles/Drones/', 'labels.csv')
    indices = np.arange(len(filenames))
    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(data, target, indices, random_state=0, shuffle=True)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    for i in range(0, len(X_test)):
        prediction = clf.predict(X_test[i].reshape(1, -1))
        print(f" The file '{filenames[idx2[i]]} is {str(prediction)}")

    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy =  {str(accuracy)}")


if __name__ == '__main__':
    main()
