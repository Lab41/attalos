from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six

from dataset_prep import DatasetPrep, RecordMetadata, SplitType

# from the tab-delimited YFCC100M data fields
META_FIELDS = [
    "photo_id",
    "user_id",
    "username",
    "date_taken",
    "upload_time",
    "camera_type",
    "title",
    "description",
    "user_tags",
    "machine_tags",
    "longitude",
    "latitude",
    "accuracy",
    "page_url",
    "download_url",
    "license",
    "license_url",
    "server",
    "farm",
    "secret",
    "original",
    "extension",
    "image_or_video"
]

class YFCC100MDatasetPrep(DatasetPrep):
    """Create a Python object to iterate over the YFCC100M data.
    This class will download missing data, but will not download it if the data
    already exists in the target directory. It will then provide an iterator
    over all of the images and their corresponding metadata within the dataset.
    Args:
        dataset_directory (str): The location to save the raw data to, or the
            location where it already exists.
        split (optional[str]): One of 'train', 'test', or 'val'. Iterates over
            just the training, test, or validation data. Defaults to 'train'.
            Currently not implemented.
    Attributes:
        split (str): One of 'train', 'test', or 'val'; indicates what split the
            iterator will return.
        item_info (dict): A dictionary mapping uniq_ids to a RecordMetadata
            object.
    """

    def __init__(self, dataset_directory, split='train'):
        super(YFCC100MDatasetPrep, self).__init__("YFCC100M", dataset_directory)
        # Set data to iterate over
        if split.lower() == 'train':
            self.split = SplitType.TRAIN
        elif split.lower() == 'test':
            self.split = SplitType.TEST
        elif split.lower() == 'val':
            raise NotImplementedError('Split type not yet implemented')
        else:
            raise NotImplementedError('Split type not yet implemented')
        # Append the filename to the full path
        self.download_dataset()

    def __remove_images_without_tags(self):
        """ Delete info about files that don't have tags. This also removes files from the wrong split"""
        keys_to_delete = []
        for f_id in self.item_info:
            if len(self.item_info[f_id]['tags']) == 0:
                keys_to_delete.append(f_id)
        for f_id in keys_to_delete:
            del self.item_info[f_id]

    def download_dataset(self):
        """
        Dataset must already be downloaded, but this integrates metadata loading and image downloading
        """
        self.item_info = {}
        # uncomment this if the filename is overall YFCC100M file, rather than a batch of it
        # for subset in os.listdir(self.data_filename):
        #with open(os.path.join(self.data_filename, subset), "r") as file:
        with open(self.dataset_directory, "r") as file:
            count = 0
            for i, line in enumerate(file):
                tokens = [item.strip() for item in line.split("\t")]
                self.download_image(tokens)

        self.__remove_images_without_tags()

    def download_image(self, tokens):
        mapping = dict(zip(META_FIELDS, tokens))
        if mapping['image_or_video'] == '0':
            key = mapping['photo_id']
                        
            image_url = mapping['download_url']

            fname = key

            # May need to add an extension to this
            self.download_if_not_present(fname, image_url)

            self.item_info[key] = {'id': key, 'fname': fname, 'tags': mapping['user_tags'].split(','), 'captions': mapping['description']}

    def get_key(self, key):
        """
        Get the description of that record by key
        Args:
            key: key
        Returns:
            (image file name, caption, tags): Returns image file name, caption string, list of tag strings
        """
        item = self.item_info[key]

        return RecordMetadata(id=key, image_name=item['fname'], tags=item['tags'], captions=item['captions'])

    def extract_image_by_key(self, key):
        """
        Extract the image from the downloaded data by key
        Args:
            key: record key t
        Returns:
            blob: Image file contents
        """
        key_info = self.get_key(key)

        fname = key_info['image_name']
        return open(fname, 'r')

    def extract_image_to_location(self, key, location):
        """
        Extract the image from the downloaded data by key and write to file location
        Args:
            key: record key t
        """
        fOut = open(location, 'wb')
        fOut.write(self.extract_image_by_key(key))
        fOut.close()

    def __iter__(self):
        """
        Iterates over dataset return metadata about dataset, one record at a time
        Returns:
            (image file name, caption string, list of tag strings associated with next item)
        """
        for key in sorted(self.list_keys()):
            yield self.get_key(key)

        raise StopIteration()

    def list_keys(self):
        """Return a list of all the unique object ids.
        Returns:
            keys: The set of keys in this dataset
        """
        return self.item_info.keys()

