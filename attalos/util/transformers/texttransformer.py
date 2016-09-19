from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import gzip
from abc import ABCMeta
from abc import abstractmethod


class TextTransformer(object):
    __metaclass__ = ABCMeta

    def __init__(self, dictionary_file=None):
        if dictionary_file:
            self.data_mapping = self.load_data_mapping(dictionary_file)

    @abstractmethod
    def create_data_mapping(self, *args, **kwargs):
        pass

    def load_data_mapping(self, dictionary_file):
        """
        Load a dictionary from a file
        Args:
            dictionary_file:

        Returns:

        """
        if dictionary_file.endswith('.gz'):
            file_pointer = gzip.open(dictionary_file, 'r')
        else:
            file_pointer = open(dictionary_file, 'r')
        data_mapping = json.load(file_pointer)
        file_pointer.close()
        return data_mapping

    def save_data_mapping(self, dictionary_file):
        """
        Save the stored dictionary to a file
        Args:
            dictionary_file:

        Returns:

        """
        if dictionary_file.endswith('.gz'):
            file_pointer = gzip.open(dictionary_file, 'w')
        else:
            file_pointer = open(dictionary_file, 'w')
        json.dump(self.data_mapping, file_pointer)
        file_pointer.close()

    def keys(self):
        return self.data_mapping.keys()

    def __getitem__(self, item):
        return self.data_mapping[item]

    def __next__(self):
        for key in self.data_mapping:
            yield key

        raise StopIteration()
