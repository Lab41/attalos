from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import zipfile
from collections import defaultdict
from attalos.dataset.dataset_prep import DatasetPrep, RecordMetadata, SplitType


TRAIN_VAL_INSTANCES_2014_URL = 'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip'
TRAIN_VAL_IMAGE_CAPTIONS_2014_URL = 'http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip'
TRAIN_IMAGE_2014_URL = 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
TEST_IMAGE_2014_URL = 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'

class MSCOCODatasetPrep(DatasetPrep):
    def __init__(self, dataset_directory, split='train'):
        """
        Initialize MS COCO specific dataset prep iterator
        Args:
            dataset_directory: Directory to store image files in
            split: Train/Val split is allowed
        Returns:

        """
        super(MSCOCODatasetPrep, self).__init__('MS COCO', dataset_directory)
        if split.lower() == 'train':
            self.split = SplitType.TRAIN
        elif split.lower() == 'test':
            self.split = SplitType.TEST
        elif split.lower() == 'val':
            raise NotImplementedError('Split type not yet implemented')
        else:
            raise NotImplementedError('Split type not yet implemented')
        self.instances_filename = self.get_candidate_filename(TRAIN_VAL_INSTANCES_2014_URL)
        self.caption_filename = self.get_candidate_filename(TRAIN_VAL_IMAGE_CAPTIONS_2014_URL)
        self.train_image_filename = self.get_candidate_filename(TRAIN_IMAGE_2014_URL)
        self.test_image_filename = self.get_candidate_filename(TEST_IMAGE_2014_URL)
        self.download_dataset()
        self.item_info = self.load_metadata()
        self.image_file_handle = None

    def download_dataset(self):
        """
        Downloads the dataset if it's not already present in the download directory
        Returns:

        """
        self.download_if_not_present(self.instances_filename, TRAIN_VAL_INSTANCES_2014_URL)
        self.download_if_not_present(self.caption_filename, TRAIN_VAL_IMAGE_CAPTIONS_2014_URL)
        self.download_if_not_present(self.train_image_filename, TRAIN_IMAGE_2014_URL)
        self.download_if_not_present(self.test_image_filename, TEST_IMAGE_2014_URL)

    def load_metadata(self):
        """
        Load the MS COCO dataset to allow for efficient iteration
        Returns:

        """
        if self.split == SplitType.TRAIN:
            split_name = 'train'
        elif self.split == SplitType.TEST:
            split_name = 'val'
        else:
            raise NotImplementedError('Split type not yet implemented')

        caption_json_fname = 'annotations/captions_%s2014.json'%split_name
        with zipfile.ZipFile(self.caption_filename) as input_file:
            item_info = {}
            train_captions = input_file.open(caption_json_fname)
            caption_info = json.loads(train_captions.read().decode("ascii"))
            for caption in caption_info['images']:
                item_info[caption['id']] = {'fname': caption['file_name'],
                                                 'id': caption['id'],
                                                 'tags': [],
                                                 'captions': []}

            for caption in caption_info['annotations']:
                item_info[caption['image_id']]['captions'].append(caption['caption'])

            del caption_info

        instance_json_fname = 'annotations/instances_%s2014.json'%split_name
        with zipfile.ZipFile(self.instances_filename) as input_file:
            train_captions = input_file.open(instance_json_fname)
            caption_info = json.loads(train_captions.read().decode("ascii"))
            category_lookup = {}
            for category in caption_info['categories']:
                category_lookup[category['id']] = category['name']

            image_tags = defaultdict(list)
            for annotation in caption_info['annotations']:
                image_tags[annotation['image_id']].append(category_lookup[annotation['category_id']])

            for image_id in image_tags:
                item_info[image_id]['tags'] = list(set(image_tags[image_id]))
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
            if self.split == SplitType.TRAIN:
                self.image_file_handle = zipfile.ZipFile(self.train_image_filename)
            elif self.split == SplitType.TEST:
                self.image_file_handle = zipfile.ZipFile(self.test_image_filename)

        if self.split ==SplitType.TRAIN:
            train_captions = self.image_file_handle.open('train2014/%s'%key_info.image_name)
            return train_captions.read()
        elif self.split ==SplitType.TEST:
            test_captions = self.image_file_handle.open('val2014/%s'%key_info.image_name)
            return test_captions.read()


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
        for key in sorted(self.list_keys()):
            yield self.get_key(key)

        raise StopIteration()

    def list_keys(self):
        """
        List all keys in the dataset
        Returns:

        """
        return self.item_info.keys()
