from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import tarfile
from attalos.dataset.dataset_prep import DatasetPrep, RecordMetadata, SplitType

# The real path to the raw data can be obtained by filling out and submitting
#  the agreement and disclaimer form at: http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm
NUSWIDE_URL = 'https:///not/the/rea/path/Flickr.tar'


class NUSWideDatasetPrep(DatasetPrep):
    def __init__(self, dataset_directory, split='train', split_division=.9):
        """
        Initialize NUS WIDE specific dataset prep iterator
        Args:
            dataset_directory: Directory to store image files in
            split: Train/Val split is allowed
        Returns:

        """
        super(NUSWideDatasetPrep, self).__init__('NUS Wide', dataset_directory)
        if split.lower() == 'train':
            self.split = SplitType.TRAIN
        elif split.lower() == 'test':
            self.split = SplitType.TEST
        elif split.lower() == 'val':
            raise NotImplementedError('Split type not yet implemented')
        else:
            raise NotImplementedError('Split type not yet implemented')
        self.data_filename = self.get_candidate_filename(NUSWIDE_URL)
        self.image_filename = None
        self.download_dataset()
        self.image_file_handle = None
        self.split_division = split_division
        self.item_info = self.load_metadata()

    def download_dataset(self):
        """
        Downloads the dataset if it's not already present in the download directory
        Returns:

        """
        self.download_if_not_present(self.data_filename, NUSWIDE_URL)
        self.image_filename = self.get_candidate_filename(NUSWIDE_URL)

    def load_metadata(self):
        """
        Load the ESP Game Metadata to allow for efficient iteration
        Returns:

        """
        if self.split == SplitType.TRAIN:
            split_name = 'train'
        elif self.split == SplitType.TEST:
            split_name = 'test'
        else:
            raise NotImplementedError('Split type not yet implemented')

        if self.image_file_handle is None:
            self.image_file_handle = tarfile.open(self.data_filename)

        item_info = {}
        for tfile in self.image_file_handle:
            if tfile.name.endswith('.jpg'):
                image_id = tfile.name
                tags = [image_id.split('/')[1]]

                # Get hash and use to partition
                m = hashlib.md5()
                m.update(image_id)
                reproducible_rand = int(m.hexdigest()[-2:], base=16)/255


                if (reproducible_rand < self.split_division and split_name == 'train') or \
                    (reproducible_rand > self.split_division and split_name == 'test'):
                    image_filename = image_id

                    item_info[image_id] = {'fname': image_filename,
                                             'id': image_id,
                                             'tags': tags,
                                             'captions': []}
        return item_info

    def get_key(self, key):
        """
        Return metadata about key
        Args:
            key: ID who's metadata we'd like to extract

        Returns:
            RecordMetadata: Returns ParserMetadata object containing metadata about item
        """
        item = self.item_info[key]
        return RecordMetadata(id=key, image_name=item['fname'], tags=item['tags'], captions=item['captions'])

    def extract_image_by_key(self, key):
        """
        Return an image based on the input key
        Args:
            key: ID of the file we'd like to extract

        Returns:
            Image Blob: Bytes of the image associated with the input ID
        """
        key_info = self.get_key(key)

        if self.image_file_handle is None:
            self.image_file_handle = tarfile.open(self.data_filename)

        train_captions = self.image_file_handle.extractfile(key_info.image_name)
        return train_captions.read()

    def extract_image_to_location(self, key, desired_file_path):
        """
        Write image based on the input key to the desired location
        Args:
            key: ID of the file we'd like to extract
            desired_file_path: Output filename that we should write the file to

        Returns:

        """
        fOut = open(desired_file_path, 'wb')
        fOut.write(self.extract_image_by_key(key))
        fOut.close()

    def __iter__(self):
        """
        Iterator over the dataset.
        Returns:
            RecordMetadata: Information about the next key
        """
        for key in self.list_keys():
            yield self.get_key(key)

        raise StopIteration()

    def list_keys(self):
        """
        List all keys in the dataset
        Returns:

        """
        if self.image_file_handle is None:
            self.image_file_handle = tarfile.open(self.data_filename)
        ordered_keys = [tfile.name for tfile in self.image_file_handle if tfile.name.endswith('.jpg')]
        return [key for key in ordered_keys if key in self.item_info]
