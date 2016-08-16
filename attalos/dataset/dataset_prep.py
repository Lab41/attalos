from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from collections import namedtuple
from abc import ABCMeta
from abc import abstractmethod

from six.moves import urllib

RecordMetadata = namedtuple('ParserMetadata', ['id', 'image_name', 'tags', 'captions'])


class SplitType:
    TRAIN = 1
    TEST = 2
    VAL = 3

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

class DatasetPrep(object):
    """ A base class for attalos data preprocessing"""
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name, dataset_directory):
        """

        Args:
            dataset_name: The name of the dataset
            dataset_directory: Directory to put downloaded dataaset into
        Returns:

        """
        self.dataset_name = dataset_name
        self.dataset_directory = dataset_directory

    @abstractmethod
    def download_dataset(self):
        """
        Args:

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
    def __iter__(self):
        """
        Iterates over dataset return metadata about dataset, one record at a time
        Returns:
            (image file name, caption string, list of tag strings associated with next item)
        """
        pass

    @abstractmethod
    def list_keys(self):
        """

        Returns:
            keys: The set of keys in this dataset
        """
        pass

    @staticmethod
    def download_if_not_present(candidate_filename, url):
        if os.path.exists(candidate_filename):
            return True
        else:
            # Taken from: http://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
            # Download the file from `url` and save it locally under `file_name`:
            filename = os.path.basename(candidate_filename)
            print('Downloading {} starting'.format(filename))
            urllib.request.urlretrieve(url, candidate_filename, reporthook)
            print('Downloading {} finished'.format(filename))

    def get_candidate_filename(self, url):
        """
        Extract the filename the file pointed at by the URL would have if
        it is already present on the file system
        Args:
            url: URL to download the file from

        Returns:

        """
        filename = os.path.basename(url)
        full_filename = os.path.join(self.dataset_directory, filename)
        return full_filename
