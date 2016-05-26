from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod



class DatasetPrep(object):
    """ A base class for attalos data preprocessing"""
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name):
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
    def extract_image_to_location(self, key, location):
        """
        Extract the image from the downloaded data by key and write to file location
        Args:
            key: record key t

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
        return self.dataset_keys()
