from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request
from collection import namedtuple
from abc import ABCMeta
from abc import abstractmethod

RecordMetadata = namedtuple('ParserMetadata', ['id', 'image_name', 'tags', 'captions'])

class DatasetPrep(object):
    """ A base class for attalos data preprocessing"""
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name):
        """

        Args:
            dataset_name: The name of the dataset

        Returns:

        """
        self.dataset_name = dataset_name


    @abstractmethod
    def download_dataset(self, data_directory):
        """
        Args:
            data_directory: Directory to put downloaded dataaset into

        Returns:
        """
        pass


    @abstractmethod
    def get_key(self, key):
        """
        Get the description of that record by key

        Args:
            key: key

        Returns:
            (image file name, caption, tags): Returns image file name, caption string, list of tag strings
        """
        pass

    @abstractmethod
    def extract_image_by_key(self, key):
        """
        Extract the image from the downloaded data by key
        Args:
            key: record key t

        Returns:
            blob: Image file contents
        """
        pass

    @abstractmethod
    def extract_image_to_location(self, key, desired_file_path):
        """
        Extract the image from the downloaded data by key and write to file location
        Args:
            key: record key t
            desired_file_path: File path to write image file to
        Returns:

        """
        pass

    @abstractmethod
    def __next__(self):
        """
        Iterates over dataset return metadata about dataset, one record at a time
        Returns:
            (image file name, caption string, list of tag strings associated with next item)
        """
        pass

    def list_keys(self):
        """

        Returns:
            keys: The set of keys in this dataset
        """
        return self.dataset_keys()

    @staticmethod
    def download_if_not_present(candidate_filename, url):
        if os.path.exists(candidate_filename):
            return True
        else:
            # Taken from: http://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
            # Download the file from `url` and save it locally under `file_name`:
            print('Downloading %s'%os.path.basename(candidate_filename))
            urllib.request.urlretrieve(url, candidate_filename)