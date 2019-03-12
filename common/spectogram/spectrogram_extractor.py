"""
The SpectrogramExtractor crawls the base folder and all its sub folder for wav files that
correspond with the valid speakers and extracts the spectrogram's of those files.

Based on previous work of Gerber, Lukic and Vogt.
"""
import os

import common.spectogram.spectrogram_converter as spectrogram_converter


class SpectrogramExtractor:

    def extract_speaker_data(self, X, y, speaker_files):
        """
        Extract spectrogram and speaker names from given folder.

        :param X: return Array that saves the mel spectrogram's
        :param y: return Array that saves the speaker numbers
        :param speaker_files: dict with speaker names as keys and audio files in an array as values
        :return: the filled X, y and the speaker names
        """

        speaker_names = []
        global_idx = 0
        curr_speaker_num = -1
        old_speaker = ''
        max_speakers = len(speaker_files.keys())

        # Crawl the base and all sub folders
        for speaker, files in enumerate(speaker_files):
            # Check files
            for full_path in files:
                # Extract speaker
                if speaker != old_speaker:
                    curr_speaker_num += 1
                    old_speaker = speaker
                    speaker_names.append(speaker)
                    print('Extraction progress: %d/%d' % (curr_speaker_num + 1, max_speakers))

                if curr_speaker_num < max_speakers:
                    extract_mel_spectrogram(full_path, X, y, global_idx, curr_speaker_num)
                    global_idx += 1

        return X[0:global_idx], y[0:global_idx], speaker_names

def extract_mel_spectrogram(wav_path, X, y, index, curr_speaker_num):
    """
    Extracts the mel spectrogram into the X array and saves the speaker into y.

    :param wav_path: the path to the wav file
    :param X: return Array that saves the mel spectrogram
    :param y: return Array that saves the speaker numbers
    :param index: the index in X and y this is stored in
    :param curr_speaker_num: the speaker number of the current speaker
    :return: a one (1) to increase the index
    """
    print(wav_path)
    return
    Sxx = spectrogram_converter.mel_spectrogram(wav_path)
    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            X[index, 0, i, j] = Sxx[i, j]
    y[index] = curr_speaker_num


# Extracts the spectrogram and discards all padded data
def extract_spectrogram(spectrogram, segment_size, frequency_elements):
    zeros = 0

    for x in spectrogram[0]:
        if x == 0.0:
            zeros += 1
        else:
            zeros = 0

    while spectrogram.shape[1] - zeros < segment_size:
        zeros -= 1

    return spectrogram[0:frequency_elements, 0:spectrogram.shape[1] - zeros]
