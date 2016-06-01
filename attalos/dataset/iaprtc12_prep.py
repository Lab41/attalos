from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lxml import objectify
import numpy as np
import os
import struct
import tarfile

import six

from dataset_prep import DatasetPrep, RecordMetadata, SplitType


IAPRTC12_URL = "http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz"
INRIA_LEAR_URL = "https://lear.inrialpes.fr/people/guillaumin/data/iccv09/iaprtc12.20091111.tar.bz2"


class Annotation(object):
    """Create a Python object from the IAPRTC12 XML annotation files.

    Args:
        xml_str: The contents of the XML file as a string.

    Attributes:
        uniq_id (str): The unique id of the annotation.
        title (str): The title of the image.
        description (str): The caption text of the image.

    """

    def __init__(self, xml_str):
        # Read the file object
        decoded = xml_str.decode("latin-1")
        coded = decoded.encode("utf-8")
        self.__tree = objectify.fromstring(coded)

        # Provide nicer accessors
        self.title = self.__tree.TITLE.text
        self.description = self.__tree.DESCRIPTION.text
        self.uniq_id = self.__tree.IMAGE.text[7:-4]


class IAPRTC12DatasetPrep(DatasetPrep):
    """Create a Python object to iterate over the IAPRTC12 data.

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
        image_data_url (str): The location of the image data online.
        image_tarball (str): The location of the image tarball on the local
            file system.
        tag_data_url (str): The location of the tag data online.
        tag_tarball (str): The location of the tag tarball on the local file
            system.
        item_info (dict): A dictionary mapping uniq_ids to a RecordMetadata
            object.


    """

    def __init__(self, dataset_directory, split='train'):
        super(IAPRTC12DatasetPrep, self).__init__("IAPRTC12", dataset_directory)
        # Set data to iterate over
        if split.lower() == 'train':
            self.split = SplitType.TRAIN
        elif split.lower() == 'test':
            raise NotImplementedError('Split type not yet implemented')
        elif split.lower() == 'val':
            raise NotImplementedError('Split type not yet implemented')
        else:
            raise NotImplementedError('Split type not yet implemented')
        # Get the data urls and download if needed
        self.image_data_url = IAPRTC12_URL
        self.image_tarball = self.get_candidate_filename(self.image_data_url)
        self.__image_tarball_filehandle = None

        self.tag_data_url = INRIA_LEAR_URL
        self.tag_tarball = self.get_candidate_filename(self.tag_data_url)
        self.__tag_tarball_filehandle = None

        self.download_dataset()
        self.__extract_filenames()
        self.__load_metadata()

    def __load_annotations(self):
        """Read the annotations from the tarball and load them into the
        class."""
        if self.__image_tarball_filehandle is None:
            self.__open_tarball()

        # We must iterate over the tarball in order because random iteration is
        # *VERY* slow
        for f in self.__image_tarball_filehandle:
            if f.name.endswith('.eng') and "annotations_complete_eng" in f.name:
                f_id = self.get_id_from_path(f.name)
                f_obj = self.__image_tarball_filehandle.extractfile(f)
                xml_str = f_obj.read()
                ann = Annotation(xml_str)
                # There are a few missing images/annotations, so we skip those cases
                try:
                    self.item_info[f_id]["captions"] = [ann.description]
                except KeyError:
                    continue

    def __load_tags(self):
        """Read the tags from the tag tarball and load them into the class."""
        if self.__tag_tarball_filehandle is None:
            self.__open_tarball()

        # Load the ID files so that we know what tags are on the various images
        id_files = {
            "train": "iaprtc12_train_list.txt",
            "test": "iaprtc12_test_list.txt",
        }
        ids = {}
        for split, file in six.iteritems(id_files):
            f = self.__tag_tarball_filehandle.extractfile(file)
            ids[split] = f.read().split()

        # Load the dictionary so that we can translate between the encoding and the words
        f = self.__tag_tarball_filehandle.extractfile("iaprtc12_dictionary.txt")
        word_map = np.array(f.read().split())

        # Load the vectors
        vec_files = {
            "train": "iaprtc12_train_annot.hvecs",
            "test": "iaprtc12_test_annot.hvecs",
        }
        word_vectors = {}
        for split, file in six.iteritems(vec_files):
            f = self.__tag_tarball_filehandle.extractfile(file)
            word_vectors[split] = self.parse_LEAR_annotation_file(f)

        # Iterate over the
        for split in ids:
            uniq_ids = ids[split]
            vecs = word_vectors[split]
            for i, uniq_id in enumerate(uniq_ids):
                vec = vecs[i]
                # There are a few missing images/annotations, so we skip those cases
                try:
                    self.item_info[uniq_id]["tags"] = list(word_map[vec == 1])
                except KeyError:
                    continue

    def __open_tarball(self):
        """ Open the tarballs and save the file handles, but only if it is not
        already open."""
        if self.__image_tarball_filehandle is None:
            self.__image_tarball_filehandle = tarfile.open(self.image_tarball, 'r')

        if self.__tag_tarball_filehandle is None:
            self.__tag_tarball_filehandle = tarfile.open(self.tag_tarball, 'r')

    def __extract_filenames(self):
        """Extract a list of the image files from the tarball."""
        if self.__image_tarball_filehandle is None:
            self.__open_tarball()

        all_files = self.__image_tarball_filehandle.getnames()

        # Extract image file names
        self.image_files = {}
        for f in all_files:
            if f.endswith('.jpg'):
                f_id = self.get_id_from_path(f)
                self.image_files[f_id] = f

    def __load_metadata(self):
        """Load the filenames, ids, tags, and captions into the metadata
        structure."""
        self.item_info = {}
        for f_id, fname in six.iteritems(self.image_files):
            self.item_info[f_id] = {
                'fname': fname,
                'id': f_id,
                'tags': [],
                'captions': [],
            }

        self.__load_annotations()
        self.__load_tags()

    def download_dataset(self):
        """
        Downloads the dataset if it's not already present in the download directory

        Returns:
        """
        self.download_if_not_present(self.image_tarball, self.image_data_url)
        self.download_if_not_present(self.tag_tarball, self.tag_data_url)

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

        if self.__image_tarball_filehandle is None:
            self.__open_tarball()

        image = self.__image_tarball_filehandle.extractfile(key_info.image_name)
        return image.read()

    def extract_image_to_location(self, key, location):
        """
        Extract the image from the downloaded data by key and write to file location

        Args:
            key: record key t

        """
        fOut = open(desired_file_path, 'wb')
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
        """

        Returns:
            keys: The set of keys in this dataset
        """
        return self.dataset_keys()

    @staticmethod
    def get_id_from_path(path):
        """Return the unique id of an image or annotation from its filepath.

        Args:
            path: The location of the file.

        Returns:
            str: The unique id

        """
        head, tail = os.path.split(path)
        _, first = os.path.split(head)
        second, _ = os.path.splitext(tail)
        return first + '/' + second

    @staticmethod
    def parse_LEAR_annotation_file(f_obj):
        """Unpack the custom LEAR data stuctures.

        Args:
            f_obj (Tarfile TarInfo object): A TarInfo object of the annotation file.

        Returns:
            numpy matrix: A two dimensional numpy array. Each row is an image,
                and each entry in the row indicates if that tag applies to the
                image.
        """
        item_size = 2  # Number of bytes for a short

        dimension = struct.unpack('h', f_obj.read(2))[0]
        num_rows = int(f_obj.size / (item_size + item_size*dimension))

        data = np.zeros((num_rows, dimension))

        data[0, :] = np.array(list(struct.unpack('{}h'.format(dimension), f_obj.read(item_size*dimension))))
        for i in range(1, num_rows):
            row_dimension = struct.unpack('h', f_obj.read(2))[0]
            if row_dimension != dimension:
                raise ValueError('Unexpected dimension: got {} expected {}'.format(row_dimension, dimension))
            data[i, :] = np.array(list(struct.unpack('{}h'.format(dimension), f_obj.read(item_size*dimension))))

        return data
