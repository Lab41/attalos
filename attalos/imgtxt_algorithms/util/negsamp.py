from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, Iterable
import numpy as np

import attalos.util.log.log as l
logger = l.getLogger(__name__)

class NegativeSampler:

    def __init__(self, word_count, fastmode=False, numsamples=1000000):
        """Initialize NegativeSampler with the word count of the corpus.
        Args:
            word_count (list): A list of numbers indicating the number of times
                a word appears in the corpus. Each location in the list
                corrisponds to a unique word.
        """
        # Set the probability distribution of labels
        self.labelpdf = word_count / word_count.sum()
        self.vocabsize = len(self.labelpdf)
        self.fastmode = fastmode
        if self.fastmode:
            logger.info('Running in fast negative sampling mode')
            self.rand_reserve = np.random.choice(self.vocabsize, 
                                                 size=numsamples, 
                                                 p=self.labelpdf)
            logger.info('Complted random reserve creation')
    
    @staticmethod
    def get_wordcount_from_datasets(datasets, one_hot):
        word_counts = np.zeros(one_hot.vocab_size)
        if isinstance(datasets, Iterable):
            iterable_datasets = datasets
        else:
            iterable_datasets = [datasets] 
        
        word_counts_text = defaultdict(int)
        for dataset in iterable_datasets:
            for tags in dataset.text_feats.values():
                for tag in tags:
                    word_counts_text[tag] += 1
        for word in word_counts_text:
            if word in one_hot:
                word_ind = one_hot.get_index(word)
                word_counts[word_ind] = word_counts_text[word]
        return word_counts
    
    def getpdf(self):
        return self.labelpdf

    def negsamp(self, vector, num2samp):
        # Defined as sampling vocabulary with known probability, discarding
        # any item that appears in test vector as a 1

        # Inverse of vector (i.e., all negative samples), saved as float
        nvector = (1-vector).astype('float32')

        # Create new probability vector excluding positive samples
        nlabelpdf = self.labelpdf * nvector
        nlabelpdf /= nlabelpdf.sum()
        
        # Negatively sample, based on probability vector
        return np.random.choice(self.vocabsize, size=num2samp, p=nlabelpdf)
    
    def negsamp_ind(self, ignored_inds, num2samp):
        if self.fastmode:
            rand_inds = np.random.randint(0, self.rand_reserve.shape[0], num2samp)
            return self.rand_reserve[rand_inds]
        nlabelpdf = np.copy(self.labelpdf)
        nlabelpdf[ignored_inds] = 0
        nlabelpdf /= nlabelpdf.sum()
        return np.random.choice(self.vocabsize, size=num2samp, p=nlabelpdf)
    
    def negsampv(self, matrix, num2samp):
        """ Vectorized negative sampler, returning multiple negatives samples
        as a multi-hot vectors after taking an input 0/1 truth matrix. As
        opposed to 'negsamp()', this function is designed for batch processing.
        Defined as sampling vocabulary with known probability (provided at
        initialization), discarding any item that appears in test vector as a 1
        Args:
            vector: multi-hot input vector/matrix: N x d
            num2samp: number of negative samples per batch
        """
        # Inverse of vector (i.e., all negative samples), saved as float
        nvector = (1-matrix).astype('float32')

        # Create new probability vector excluding positive samples
        nlabelpdf = self.labelpdf * nvector
        nlabelpdf = (nlabelpdf.T/(nlabelpdf.sum(axis=1))).T

        # Negatively sample, based on probability vector
        nvector.fill(0)

        for i in xrange(0, nlabelpdf.shape[0]):
            nvector[i, np.random.choice(self.vocabsize, size=num2samp, p=nlabelpdf[i])] = 1

        return nvector
    
    def binxentropy(self, y, yh):
        # Defined as:
        # C = 1/N sum_i yi sig(yhi) + (1 - yi) sig( 1-yhi )
        return y.dot(sigmoid(yh)) + (1-y).dot(sigmoid(1-yh))

    def nscost(self, y, yh, num2samp):
        # Negative sampling cost function, utilizing binxentropy
        negidx = self.negsamp(y, num2samp)
        posidx = self.posidx(y)
        posneg = np.concatenate((posidx, negidx))
        return self.binxentropy(y[posneg], yh[posneg])

    def posidx(self, vector):
        # Get the indices where there are ones
        return np.arange(self.vocabsize)[vector > 0.5]

    def sigmoid(x):
        # Sigmoid function
        return 1. / (1. + np.exp(-x))
