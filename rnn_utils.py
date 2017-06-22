# coding: utf-8

import numpy as np
from keras.preprocessing.sequence import pad_sequences

def rnn_formatter(sequence,
                  start_signal=None,
                  end_signal=None,
                  dtype="int32",
                  **kwargs):
    """
    Args:
        sequence: list of integers
        start_signal: integer added to the top
        end_signal: integer added to the end
        kwargs: args passed to pad_sequences
    Returns:
        formatted numpy array for RNN
    """

    sequence = list(sequence)
    if not start_signal is None:
        sequence.insert(0, start_signal)
    if not end_signal is None:
        sequence.append(end_signal)

    X_lang = pad_sequences([sequence[0:i] for i in range(1, len(sequence))],
                           dtype=dtype, **kwargs)
    y_lang = np.array(sequence[1:], dtype=dtype)

    return X_lang, y_lang
