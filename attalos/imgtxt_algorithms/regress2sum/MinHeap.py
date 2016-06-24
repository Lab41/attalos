import numpy as np


class MinHeap:
    """
    A quick and dirty minheap implementation to test out the brute force search
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []
        self.max_acceptable = None

    def get_max(self):
        return self.data[self.get_max_index()][0]

    def get_max_index(self):
        return self.data.index(max(self.data))

    def insert(self, item, index):
        if len(self.data) < self.max_size:
            self.data.append((item, index))
            self.max_acceptable = self.get_max()
        elif item < self.max_acceptable:
            del self.data[self.get_max_index()]
            self.data.append((item, index))
            self.max_acceptable = self.get_max()

    def get_result(self):
        if self.max_size == 1:
            return self.data[0][1]
        else:
            return [v for (k, v) in self.data]


def get_distance(v1, v2):
    # Euclidean
    return np.sqrt(np.sum(np.square(np.subtract(v1,v2))))


def get_top_k(target_vector, w2v_lookup, k, distance_func=get_distance):
    """
    Brute force search of word2vec space for top K nearest neighbors
    Args:
        target_vector: The vector we are looking to find the nearest neighbors of
        w2v_lookup: A dictionary like object with the keys being words and the values being word vectors
        k: The number of nearest neighbors to find
        distance_func: Distance function to use. Should take two vectors and return a distance

    Returns:
        List of the top K words
    """
    mh = MinHeap(k)
    for word in w2v_lookup:
        mh.insert(distance_func(target_vector, w2v_lookup[word]), word)
    return mh.get_result()
