from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import zipfile
from collections import defaultdict
from dataset_prep import DatasetPrep, RecordMetadata, SplitType

import sys

VISUAL_GENOME_IMAGES = 'https://cs.stanford.edu/people/rak248/VG_100K/images.zip'
VISUAL_GENOME_METADATA = 'https://visualgenome.org/static/data/dataset/image_data.json.zip'
VISUAL_GENOME_REGIONS = 'https://visualgenome.org/static/data/dataset/region_descriptions.json.zip'
VISUAL_GENOME_OBJECTS = 'https://visualgenome.org/static/data/dataset/objects.json.zip'
VISUAL_GENOME_ATTRIBUTES = 'https://visualgenome.org/static/data/dataset/attributes.json.zip'
VISUAL_GENOME_RELATIONSHIPS = 'https://visualgenome.org/static/data/dataset/relationships.json.zip'

# Final import is the API
sys.path.append(VISUAL_GENOME_API)
import src.api as api
import src.local as lapi
import src.utils as utils
import src.models as models

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
        self.download_dataset()
	self.load_metadata()
	self.images_file_handle = None

    def download_dataset(self, extract_all_images=False):
        """
        Downloads the dataset if it's not already present in the download directory
        Returns:
        """

	self.download_if_not_present(self.metadata_filename, VISUAL_GENOME_METADATA)
	self.download_if_not_present(self.images_filename, VISUAL_GENOME_IMAGES )
        self.download_if_not_present(self.relationships_filename, VISUAL_GENOME_RELATIONSHIPS)
        self.download_if_not_present(self.objects_filename, VISUAL_GENOME_OBJECTS)
        self.download_if_not_present(self.attributes_filename, VISUAL_GENOME_ATTRIBUTES)

	import os
	if not os.path.exists(self.metadata_filename[:-4]):
		zipref = zipfile.ZipFile(self.metadata_filename,'r')
		zipref.extractall(self.data_dir)
	if not os.path.exists(self.relationships_filename[:-4]):
        	zipref = zipfile.ZipFile(self.relationships_filename,'r')
        	zipref.extractall(self.data_dir)
	if not os.path.exists(self.objects_filename[:-4]):
        	zipref = zipfile.ZipFile(self.objects_filename,'r')
        	zipref.extractall(self.data_dir)
	if not os.path.exists(self.attributes_filename[:-4]):
        	zipref = zipfile.ZipFile(self.attributes_filename,'r')
        	zipref.extractall(self.data_dir)

	if extract_all_images:
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

	self.item_info = json.loads(open(self.metadata_filename[:-4], 'r').read().decode('ascii'))
	# self.item_info=lapi.GetAllImageData(dataDir=self.data_dir)
	

    def get_key(self, key):
        """
        Return metadata about key
        Args:
            key: ID who's metadata we'd like to extract

        Returns:
            RecordMetadata: Returns ParserMetadata object containing metadata about item
        """
        
	# item = self.item_info[key]
        # return RecordMetadata(id=key, images_name=item['fname'], tags=item['tags'], captions=item['captions'])

	return self.item_info[key-1]

    def extract_image_by_key(self, key):
        """
        Return an image based on the input key
        Args:
            key: ID of the file we'd like to extract

        Returns:
            Image Blob: Bytes of the image associated with the input ID
        """
	imname = self.item_info[key].url.split('/')[-1]

        key_info = self.get_key(key)

        if self.images_file_handle is None:
            self.images_file_handle = zipfile.ZipFile(self.images_filename)

	zipfile.ZipFile.namelist(self.images_file_handle)
	print('image name %s'%(self.images_filename))
        train_captions = self.images_file_handle.open('%s'%imname)

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
            yield self.get_key(key)

        raise StopIteration()

    def list_keys(self):
        """
        List all keys in the dataset
        Returns:

        """
        return self.item_info.keys()
