from SoundProcessor import getMFCC
from scipy.io import wavfile


def main():
    file = wavfile.read('wavfiles/Violin1.wav')
    mfcc = getMFCC(file)
    print(mfcc)


if __name__ == '__main__':
    main()
