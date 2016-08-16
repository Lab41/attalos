import numpy as np


def construct_W(w2v_model, vocab, dtype=np.float64):
    """
    Creates and returns a numpy ndarray of dimensions (200 x size(vocab)).
    The nth column in this array represents the Word2Vec vector for the nth word in vocab.
    Words in vocab not in the vocabulary for w2v_model will be ignored/skipped.

    Args:
        w2v_model: a Word2Vec model from the word2vec package (available in pypi)
        vocab: a list of words

    Returns:
        a numpy ndarray
    """
    return np.asarray([w2v_model.get_vector(word) for word in vocab if word in w2v_model.vocab], dtype=dtype).T


def get_invalid_labels(labels, valid_labels):
    """
    Returns a dictionary whose keys are words in labels that are not in valid_labels.
    The values of the dictionary are the corresponding indices of the invalid words in labels.

    Args:
        labels: a list of labels
        valid_labels: a list of valid labels.

    Returns:
        a dictionary
    """
    return {word: idx for (idx, word) in enumerate(labels) if word not in valid_labels}


def broadcast_transform(arr, v):
    """
    Transforms v so that it is broadcastable to arr in a row-wise fashion.
    For instance, if v = [1, 2, 3, 4], and arr is a 3x4 numpy ndarray, then the return value is:
    [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4]
    ]

    Args:
        v: a numpy vector (1d ndarray)

    Returns:
        a numpy ndarray
    """
    return v[np.newaxis].T.repeat(arr.shape[1]).reshape(arr.shape)


def scale(arr):
    """
    Returns a new numpy ndarray where the nth row is the scaled version of the nth row of arr.
    Each row is scaled such that all values are between 0 and 1.

    Args:
        arr: a numpy ndarray

    Returns:
        a numpy ndarray
    """
    broadcast_min = broadcast_transform(arr, np.amin(arr, 1))
    broadcast_max = broadcast_transform(arr, np.amax(arr, 1))
    return (arr - broadcast_min) / (broadcast_max - broadcast_min)


def scale2(arr):
    """
    Returns a new numpy ndarray where the nth row is the scaled2 version of the nth row of arr.
    Each row is scaled such that it has mean 0 and stdev 1.

    Args:
        arr: a numpy ndarray

    Returns:
        a numpy ndarray
    """
    broadcast_mean = broadcast_transform(arr, np.mean(arr, axis=1))
    broadcast_std = broadcast_transform(arr, np.std(arr, axis=1))
    return (arr - broadcast_mean) / broadcast_std


def nonlinearity(arr, coef=-1, offset=1, power=2, alpha=0.005):
    """
    This function applies a nonlinearity to an numpy ndarray.
    The nonlinearity is meant to better separate high values from low values.
    See the Attalos documentation for additional details.

    Args:
        arr: a numpy ndarray
        coef: coefficient for multiplication
        offset: offset for arr
        power: power to raise each element to
        alpha: divisor for arr

    Returns:
        a numpy ndarray
    """
    return np.exp(coef * np.power((arr - offset), power) / alpha)


def original_value_filter(arr, original_multihot):
    """
    This function returns a copy of arr where all values are suppressed except for the ones selected by original_multihot.
    In other words, non-selected values are set to 0.

    Args:
        arr:
        original_multihot:

    Returns:
    """
    filtered_arr = np.copy(arr)
    for row_idx, row in enumerate(filtered_arr):
        for idx, val in enumerate(row):
            if idx not in np.where(original_multihot[row_idx]>0)[0]:
                row[idx] = 0
    return filtered_arr


def top_n_value_filter(arr, n, reverse=False, suppression_fn=lambda x: 0):
    """
    This function returns a copy of arr where all values except for the top n are suppressed.
    In other words, all values except for the top n are passed to the suppression function (default: set to 0).

    In the case where reverse = True, perform the same operations except use the bottom n instead of top n.

    Args:
        arr:
        n:
        reverse:

    Returns:
    """
    filtered_arr = np.zeros(arr.shape)
    for row_idx, row in enumerate(arr):
        ordered_idxs = row.argsort()[::-1]
        selected_idxs = ordered_idxs[-n:] if reverse else ordered_idxs[:n]
        for idx, val in enumerate(row):
            if idx in selected_idxs:
                filtered_arr[row_idx][idx] = val
            else:
                filtered_arr[row_idx][idx] = suppression_fn(val)
    return filtered_arr


def original_or_top_n_value_filter(arr, original_multihot, n, reverse=False, suppression_fn=lambda x: 0):
    filtered_arr = np.copy(arr)
    for row_idx, row in enumerate(filtered_arr):
        ordered_idxs = row.argsort()[::-1]
        selected_idxs = ordered_idxs[-n:] if reverse else ordered_idxs[:n]
        for idx, val in enumerate(row):
            if idx not in np.where(original_multihot[row_idx]>0)[0] and idx not in selected_idxs:
                filtered_arr[row_idx][idx] = suppression_fn(val)
    return filtered_arr
