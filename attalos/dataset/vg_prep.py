from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import zipfile
from collections import defaultdict
from dataset_prep import DatasetPrep, RecordMetadata, SplitType

import sys

# Import the Visual Genome API for functionality
VISUAL_GENOME_API = '/work/attalos/accessVG'

VISUAL_GENOME_IMAGES = 'https://visualgenome.org/static/data/dataset/image_data.json.zip'
VISUAL_GENOME_METADATA = 'https://visualgenome.org/static/data/dataset/image_data.json.zip'
VISUAL_GENOME_REGIONS = 'https://visualgenome.org/static/data/dataset/region_descriptions.json.zip'
VISUAL_GENOME_OBJECTS = 'https://visualgenome.org/static/data/dataset/objects.json.zip'
VISUAL_GENOME_ATTRIBUTES = 'https://visualgenome.org/static/data/dataset/attributes.json.zip'
VISUAL_GENOME_RELATIONSHIPS = 'https://visualgenome.org/static/data/dataset/relationships.json.zip'

# Final import is the API
sys.path.append(VISUAL_GENOME_API)
import src.api as api


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
        # self.download_dataset()
	self.load_metadata()

    def download_dataset(self):
        """
        Downloads the dataset if it's not already present in the download directory
        Returns:

        """
        self.download_if_not_present(self.metadata_filename, VISUAL_GENOME_METADATA)
        self.download_if_not_present(self.images_filename, VISUAL_GENOME_IMAGES )
        self.download_if_not_present(self.relationships_filename, VISUAL_GENOME_RELATIONSHIPS)
        self.download_if_not_present(self.objects_filename, VISUAL_GENOME_OBJECTS)
        self.download_if_not_present(self.attributes_filename, VISUAL_GENOME_ATTRIBUTES)
       
	zipref = zipfile.ZipFile(self.metadata_filename,'r')
	zipref.extractall(data_dir)
	zipref = zipfile.ZipFile(self.images_filename,'r')
	zipref.extractall(data_dir)
        zipref = zipfile.ZipFile(self.relationships_filename,'r')
        zipref.extractall(data_dir)
        zipref = zipfile.ZipFile(self.objects_filename,'r')
        zipref.extractall(data_dir)
        zipref = zipfile.ZipFile(self.attributes_filename,'r')
        zipref.extractall(data_dir)
	

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

	self.item_info=api.GetAllImageIds()

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

	return api.GetImageIdsInRange(startIndex=key, endIndex=key+1)

    def extract_image_by_key(self, key):
        """
        Return an image based on the input key
        Args:
            key: ID of the file we'd like to extract

        Returns:
            Image Blob: Bytes of the image associated with the input ID
        """
	NotImplementedError('Split type not yet implemented')

    def extract_image_to_location(self, key, desired_file_path):
        """
        Write image based on the input key to the desired location
        Args:
            key: ID of the file we'd like to extract
            desired_file_path: Output filename that we should write the file to

        Returns:

        """
	raise NotImplementedError('Split type not yet implemented')

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
