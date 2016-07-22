from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import shutil
from attalos.dataset.dataset_prep import DatasetPrep, RecordMetadata, SplitType


class GenericDatasetPrep(DatasetPrep):
    def __init__(self, dataset_description, split='train'):
        """
        Initialize generic dataset prep
        Args:
            dataset_description: File containing the description of the dataset
            split: Train/Val split is allowed
        Returns:

        """
        super(GenericDatasetPrep, self).__init__('GenericDataset', dataset_description)
        if split.lower() == 'train':
            self.split = SplitType.TRAIN
        elif split.lower() == 'test':
            self.split = SplitType.TEST
        elif split.lower() == 'val':
            raise NotImplementedError('Split type not yet implemented')
        else:
            raise NotImplementedError('Split type not yet implemented')
        self.dataset_description = dataset_description
        self.download_dataset()
        self.item_info = self.load_metadata()
        self.image_file_handle = None

    def download_dataset(self):
        """
        Downloads the dataset if it's not already present in the download directory
        Returns:

        """
        pass

    def load_metadata(self):
        """
        Load the MS COCO dataset to allow for efficient iteration
        Returns:

        """
        if self.split == SplitType.TRAIN:
            split_name = 'train'
        elif self.split == SplitType.TEST:
            split_name = 'test'
        else:
            raise NotImplementedError('Split type not yet implemented')

        if self.dataset_description.endswith('.gz'):
            open_fn = gzip.open
        else:
            open_fn = open

        I_FNAME = 0
        I_SPLIT = 1
        I_TAGS = 2
        item_info = {}
        with open_fn(self.dataset_description, 'rt') as input_file:
            for line in input_file:
                ls = line.strip().split('\t')
                if ls[I_SPLIT].lower() == split_name:
                    item_info[ls[I_FNAME]] = {'fname': ls[I_FNAME],
                                             'id': ls[I_FNAME],
                                             'tags': ls[I_TAGS].split(','),
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

        train_captions = open(key_info.image_name)
        return train_captions.read()

    def extract_image_to_location(self, key, desired_file_path):
        """
        Write image based on the input key to the desired location
        Args:
            key: ID of the file we'd like to extract
            desired_file_path: Output filename that we should write the file to

        Returns:

        """
        key_info = self.get_key(key)
        shutil.copy(key_info.image_name, desired_file_path)

    def __iter__(self):
        """
        Iterator over the dataset.
        Returns:
            RecordMetadata: Information about the next key
        """
        for key in sorted(self.list_keys()):
            yield self.get_key(key)

        raise StopIteration()

    def list_keys(self):
        """
        List all keys in the dataset
        Returns:

        """
        return self.item_info.keys()
