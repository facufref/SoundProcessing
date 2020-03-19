from SoundProcessor import get_processed_mfcc, get_processed_filter_banks
from scipy.io import wavfile


def main():
    file = wavfile.read('wavfiles/Violin1.wav')
    mfcc = get_processed_mfcc(file)
    print(mfcc)
    filter_banks = get_processed_filter_banks(file)
    print(filter_banks)


if __name__ == '__main__':
    main()
