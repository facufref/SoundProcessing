from SoundProcessor import get_mfcc
from scipy.io import wavfile


def main():
    file = wavfile.read('wavfiles/Violin1.wav')
    mfcc = get_mfcc(file)
    print(mfcc)


if __name__ == '__main__':
    main()
