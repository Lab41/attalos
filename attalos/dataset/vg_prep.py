from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import zipfile
from collections import defaultdict
from attalos.dataset.dataset_prep import DatasetPrep, RecordMetadata, SplitType

import sys
import os

VISUAL_GENOME_IMAGES = 'https://cs.stanford.edu/people/rak248/VG_100K/images.zip'
VISUAL_GENOME_METADATA = 'https://visualgenome.org/static/data/dataset/image_data.json.zip'
VISUAL_GENOME_REGIONS = 'https://visualgenome.org/static/data/dataset/region_descriptions.json.zip'
VISUAL_GENOME_OBJECTS = 'https://visualgenome.org/static/data/dataset/objects.json.zip'
VISUAL_GENOME_ATTRIBUTES = 'https://visualgenome.org/static/data/dataset/attributes.json.zip'
VISUAL_GENOME_RELATIONSHIPS = 'https://visualgenome.org/static/data/dataset/relationships.json.zip'

# Wrapper for the Visual Genome
class VGDatasetPrep(DatasetPrep):
    def __init__(self, dataset_directory, split='train'):
        """
        Initialize Visual Genome specific dataset prep iterator
        Args:
            dataset_directory: Directory to store image files in
            split: Train/Val split is allowed
        Returns:

        """
        super(VGDatasetPrep, self).__init__('Visual Genome ', dataset_directory)

        if split.lower() == 'train':
            self.split = SplitType.TRAIN
        elif split.lower() == 'test':
            raise NotImplementedError('Split type not yet implemented')
        elif split.lower() == 'val':
            raise NotImplementedError('Split type not yet implemented')
        else:
            raise NotImplementedError('Split type not yet implemented')

        self.data_dir = dataset_directory

        self.relationships_filename = self.get_candidate_filename(VISUAL_GENOME_RELATIONSHIPS)
        self.metadata_filename = self.get_candidate_filename(VISUAL_GENOME_METADATA)
        self.images_filename = self.get_candidate_filename(VISUAL_GENOME_IMAGES)
        self.objects_filename = self.get_candidate_filename(VISUAL_GENOME_OBJECTS)
        self.attributes_filename = self.get_candidate_filename(VISUAL_GENOME_ATTRIBUTES)
        self.regions_filename = self.get_candidate_filename(VISUAL_GENOME_REGIONS)
        self.download_dataset()
        self.load_metadata()
        self.images_file_handle = None

    def download_dataset(self, extract_files=False):
        """
        Downloads the dataset if it's not already present in the download directory
        Returns:
        """

        self.download_if_not_present(self.metadata_filename, VISUAL_GENOME_METADATA)
        self.download_if_not_present(self.images_filename, VISUAL_GENOME_IMAGES )
        self.download_if_not_present(self.relationships_filename, VISUAL_GENOME_RELATIONSHIPS)
        self.download_if_not_present(self.objects_filename, VISUAL_GENOME_OBJECTS)
        self.download_if_not_present(self.attributes_filename, VISUAL_GENOME_ATTRIBUTES)
        self.download_if_not_present(self.regions_filename, VISUAL_GENOME_REGIONS)

        if not os.path.exists(self.metadata_filename[:-4]) and extract_files:
            zipref = zipfile.ZipFile(self.metadata_filename,'r')
            zipref.extractall(self.data_dir)
        if not os.path.exists(self.relationships_filename[:-4]) and extract_files:
            zipref = zipfile.ZipFile(self.relationships_filename,'r')
            zipref.extractall(self.data_dir)
        if not os.path.exists(self.objects_filename[:-4]) and extract_files:
            zipref = zipfile.ZipFile(self.objects_filename,'r')
            zipref.extractall(self.data_dir)
        if not os.path.exists(self.attributes_filename[:-4]) and extract_files:
            zipref = zipfile.ZipFile(self.attributes_filename,'r')
            zipref.extractall(self.data_dir)
        if not os.path.exists(self.images_filename[:-4]) and extract_files:
            zipref = zipfile.ZipFile(self.images_filename,'r')
            zipref.extractall(self.data_dir)

    def load_metadata(self):
        """
        Load the VGG dataset to allow for efficient iteration
        Returns:

        """
        if self.split == SplitType.TRAIN:
            split_name = 'train'
        elif self.split == SplitType.VAL:
            split_name = 'val'
        else:
            raise NotImplementedError('Split type not yet implemented')

        # Load Image Metadata
        metadata_raw_name = os.path.basename(self.metadata_filename)[:-1*len('.zip')]
        json_file = zipfile.ZipFile(self.metadata_filename).open(metadata_raw_name)
        json_str = json_file.read()
        item_info = json.loads(json_str)
        self.item_keys = [ item_id['id'] for item_id in item_info ]
        self.item_info = dict(zip(self.item_keys, item_info))

        # Load object data
        objects_raw_name = os.path.basename(self.objects_filename)[:-1*len('.zip')]
        objects_data = json.loads(zipfile.ZipFile(self.objects_filename).open(objects_raw_name).read())
        self.tags_data = {}
        for row in objects_data:
            try:
                # TODO: Check if taking the first name is a reasonable thing to do
                objects = set([object['names'][0] for object in row['objects']])
            except:
                print(row)
                raise
            self.tags_data[row['id']] = list(objects)

        # Load caption data
        captions_raw_name = os.path.basename(self.regions_filename)[:-1*len('.zip')]
        objects_data = json.loads(zipfile.ZipFile(self.regions_filename).open(captions_raw_name).read())
        self.captions_data = {}
        for row in objects_data:
            try:
                # TODO: Check if taking the first name is a reasonable thing to do
                captions = set([region['phrase'] for region in row['regions']])
            except:
                print(row)
                raise
            self.captions_data[row['id']] = list(captions)


    def get_key(self, key):
        """
        Return metadata about key
        Args:
            key: ID who's metadata we'd like to extract

        Returns:
            RecordMetadata: Returns ParserMetadata object containing metadata about item
        """
        item = self.item_info[key]
        image_name = os.path.basename(item['url'])

        if key in self.tags_data:
            tags = self.tags_data[key]
        else:
            tags = None

        if key in self.tags_data:
            captions = self.captions_data[key]
        else:
            captions = None

        return RecordMetadata(id=key, image_name=image_name, tags=tags, captions=captions)

        return self.item_info[key]

    def extract_image_by_key(self, key):
        """
        Return an image based on the input key
        Args:
            key: ID of the file we'd like to extract

        Returns:
            Image Blob: Bytes of the image associated with the input ID
        """
        key_info = self.get_key(key)

        if self.images_file_handle is None:
            self.images_file_handle = zipfile.ZipFile(self.images_filename)

        train_captions = self.images_file_handle.open('%s'%format(key_info.image_name))

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

    def get_image_by_key(self, key):
        ''' 
        Gets the image (assuming it's been extracted).
        '''
        imdata = self.get_key(key)
        imurl = imdata['url'].split('/')[-1]
        return self.data_dir+imurl

    def __iter__(self):
        """
        Iterator over the dataset.
        Returns:
            RecordMetadata: Information about the next key
        """
        for key in sorted(self.list_keys()):
            if 'VG_100K' in self.item_info[key]['url']:
                yield self.get_key(key)

        raise StopIteration()

    def list_keys(self):
        """
        List all keys in the dataset
        Returns:
        """

        return self.item_info.keys()
