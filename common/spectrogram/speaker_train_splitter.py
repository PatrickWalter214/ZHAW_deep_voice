"""
Class to generate a callable that splits the input data into train and validation data.

Based on previous work of Gerber, Lukic and Vogt.
"""
import numpy as np


class SpeakerTrainSplit(object):
    """
    Parameters
    ----------
    eval_size : float
        Evaluation size of training set.
    """

    def __init__(self, eval_size):
        self.eval_size = eval_size


    # lehmacl1@2019-03-20:
    # Assumption for transition from TIMIT-coupled framework towards
    # usage of Voxceleb2 data: 1 sentence = 1 audio file
    #
    # In this method, data is being split into training and validation set.
    # Due to the nature that data per speaker does not need to be uniform,
    # it can be that we need to round the eval_size according to the number
    # of files per speaker.
    #
    # (for eval_size = 0.2)
    # We cannot assume that exactly 20% of the total number of files will end up
    # in the validation set while the training set contains exactly 80%. Instead,
    # for each speaker we round so that the ratio per speaker in the validation
    # and train set is as close to the eval_size as possible.
    #
    # IMPORTANT!!!
    # There need to be enough files per speaker to ensure that at least 1 file is in each set
    # for each speaker. E.g. for a :eval_size of 0.2, at least 5 files per speaker are encouraged.
    #
    def __call__(self, X, y):
        # Build speaker file count dict: Iterate over all entries in y (labels) and
        # sum up the amount of examples for each unique speaker_id (content of y)
        #
        speaker_file_dict = dict()

        for i in range(len(y)):
            try:
                speaker_file_dict[y[i]].append(i)
            except KeyError:
                speaker_file_dict[y[i]] = [i]

        valid_size = int(len(y) * self.eval_size)  # 0.2y - len(y) is amount of total audio files
        train_size = int(len(y) - valid_size)  # 0.8y - len(y) is amount of total audio files

        X_new = np.zeros(X.shape, dtype=X.dtype)
        y_new = np.zeros(y.shape, dtype=y.dtype)

        # To avoid unnecessary resizes for the X_train/X_valid and y_train/y_valid Arrays,
        # we instead reorder X and y by filling them into X_new and y_new with the same shapes
        # but start filling the train part from index 0 and the valid part with reverse index from
        # the end (len(y) - 1)
        #
        train_index = 0
        valid_index = len(y) - 1

        # print(X.shape)
        # print(y.shape)

        for speaker_id in speaker_file_dict.keys():
            num_files_for_speaker = len(speaker_file_dict[speaker_id])

            speaker_valid_size = int(round(num_files_for_speaker * self.eval_size, 0))
            speaker_train_size = int(num_files_for_speaker - speaker_valid_size)

            # print("Speaker {} has {} files:\ttrain{}\tvalid{}".format(speaker_id, num_files_for_speaker, speaker_train_size, speaker_valid_size))

            for i in range(num_files_for_speaker):
                dataset_index = speaker_file_dict[speaker_id][i]

                if i > speaker_train_size - 1:
                    X_new[valid_index] = X[dataset_index]
                    y_new[valid_index] = y[dataset_index]
                    valid_index -= 1

                else:
                    X_new[train_index] = X[dataset_index]
                    y_new[train_index] = y[dataset_index]
                    train_index += 1

            # print("\tSpeaker {}, y_valid: {}, y_train: {}".format(speaker_id, y_new[valid_index + 1], y_new[train_index - 1]))

        # The indices were incremented / decremented after the last step, with this correction
        # we get the last used indices for both
        #
        valid_index += 1
        train_index -= 1

        # Split the new sorted array into the train and validation parts
        #
        [X_train, X_valid] = np.split(X_new, [train_index])  # pylint: disable=unbalanced-tuple-unpacking
        [y_train, y_valid] = np.split(y_new, [train_index])  # pylint: disable=unbalanced-tuple-unpacking

        return X_train, X_valid, y_train, y_valid


class SpeakerTrainMFCCSplit(object):
    """
    Parameters
    ----------
    eval_size : float
        Evaluation size of training set.
    sentences : int
        Number of sentences in training set for each speaker.
    """

    def __init__(self, eval_size, sentences):
        self.eval_size = eval_size
        self.sentences = sentences

    def __call__(self, X, y, net=None):
        valid_size = int(len(y) * self.eval_size)
        train_size = int(len(y) - valid_size)
        X_train = np.zeros((train_size,  X[0].shape[0], X[0].shape[1]), dtype=np.float32)
        X_valid = np.zeros((valid_size,  X[0].shape[0], X[0].shape[1]), dtype=np.float32)
        y_train = np.zeros(train_size, dtype=np.int32)
        y_valid = np.zeros(valid_size, dtype=np.int32)

        train_index = 0
        valid_index = 0
        nth_elem = self.sentences - self.sentences * self.eval_size
        for i in range(len(y)):
            if i % self.sentences >= nth_elem:
                X_valid[valid_index] = X[i]
                y_valid[valid_index] = y[i]
                valid_index += 1
            else:
                X_train[train_index] = X[i]
                y_train[train_index] = y[i]
                train_index += 1

        return X_train, X_valid, y_train, y_valid
