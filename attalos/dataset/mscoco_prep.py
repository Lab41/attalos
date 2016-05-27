from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import zipfile
from collections import defaultdict
from dataset_prep import DatasetPrep, RecordMetadata


TRAIN_VAL_INSTANCES_2014_URL = 'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip'
TRAIN_VAL_IMAGE_CAPTIONS_2014_URL = 'http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip'


class MSCOCODatasetPrep(DatasetPrep):
    def __init__(self, dataset_directory):
        super(MSCOCODatasetPrep, self).__init__('MS COCO')
        self.dataset_directory = dataset_directory
        self.instances_filename = self.get_candidate_filename(TRAIN_VAL_INSTANCES_2014_URL)
        self.caption_filename = self.get_candidate_filename(TRAIN_VAL_IMAGE_CAPTIONS_2014_URL)
        self.download_dataset()
        self.item_info = self.load_metadata()

    def download_dataset(self):
        self.download_if_not_present(self.instances_filename, TRAIN_VAL_INSTANCES_2014_URL)
        self.download_if_not_present(self.caption_filename, TRAIN_VAL_IMAGE_CAPTIONS_2014_URL)

    def load_metadata(self):
        with zipfile.ZipFile(self.caption_filename) as input_file:
            item_info = {}
            train_captions = input_file.open('annotations/captions_train2014.json')
            caption_info = json.loads(train_captions.read().decode("ascii"))
            for caption in caption_info['images']:
                item_info[caption['id']] = {'fname': caption['file_name'],
                                                 'id': caption['id'],
                                                 'captions': []}

            for caption in caption_info['annotations']:
                item_info[caption['image_id']]['captions'].append(caption['caption'])

            del caption_info
        with zipfile.ZipFile(self.instances_filename) as input_file:
            train_captions = input_file.open('annotations/instances_train2014.json')
            caption_info = json.loads(train_captions.read().decode("ascii"))
            image_tags = defaultdict(list)
            for annotation in caption_info['annotations']:
                image_tags[annotation['image_id']].append(annotation['category_id'])

            for image_id in image_tags:
                item_info[image_id]['tags'] = list(set(image_tags[image_id]))
            return item_info

    def get_key(self, key):
        """
        Return metadata about key
        Args:
            key: ID who's metadata we'd like to extract

        Returns:
            ParserMetadata: Returns ParserMetadata object containing metadata about item
        """
        item = self.item_info[key]
        return RecordMetadata(id=key, image_name=item['fname'], tags=item['tags'], captions=item['captions'])


    def extract_image_by_key(self, key):
        key_info = self.get_key(key)
        with zipfile.ZipFile(self.data_fname) as input_file:
            train_captions = input_file.open('train2014/%s'%key_info.image_name)
            return train_captions.read()


    def extract_image_to_location(self, key, desired_file_path):
        fOut = open(desired_file_path, 'w')
        fOut.write(self.extract_image_by_key(key))
        fOut.close()

    def __next__(self):
        for key in sorted(self.list_keys()):
            return self.get_key(key)

        raise StopIteration()

    def get_candidate_filename(self, url):
        """
        Extract the filename from a URL
        Args:
            url:

        Returns:

        """
        filename = os.path.basename(url)
        full_filename = os.path.join(self.dataset_directory, filename)
        return full_filename

    def list_keys(self):
        return self.item_info.keys()



t = MSCOCODatasetPrep('/data/mscoco')
print(t.load_captions()[524286])


