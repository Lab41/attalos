from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from attalos.dataset.texttransformer import TextTransformer

class OneHot(TextTransformer):
    """
    Transforms tags from a dataset iterator into into a one-hot encoding
    """
    def __init__(self, dataset, dictionary_file=None):
        """
        Initialize OneHot encoding
        Args:
            dataset (attalos.dataset.dataset): A dataset iterator
            dictionary_file: A saved dictionary file

        Returns:

        """
        super(OneHot, self).__init__(dictionary_file)
        if dictionary_file:
            self.num_keys = len(super(OneHot, self).keys())
        else:
            self.data_mapping = {}
            self.create_data_mapping(dataset)

    def create_data_mapping(self, dataset):
        dataset_tags = set()
        for tags in dataset.text_feats.values():
            dataset_tags.update(tags)

        self.num_keys = len(dataset_tags)
        for i, key in enumerate(dataset_tags):
            self.data_mapping[key] = i

    def get_multiple(self, tags):
        """
        Get the multi-hot encoding for a list of tags
        Args:
            tags (list): List of tags for which to return a multi-hot encoding

        Returns:
            mulithot_feats (ndarray): Returns a multi-hot numpy array
        """
        multihot_feats = np.zeros(self.num_keys)
        for tag in tags:
            multihot_feats += self.__getitem__(tag)
        return multihot_feats

    def __getitem__(self, item):
        index = self.data_mapping[item]
        arr = np.zeros(self.num_keys)
        arr[index] = 1
        return arr


def main():
    import argparse
    from attalos.dataset.dataset import Dataset

    parser = argparse.ArgumentParser(description='Test One-Hot Encoding')
    parser.add_argument('--image_feature_file',
                      dest='image_feature_file',
                      type=str,
                      help='Image Feature file')
    parser.add_argument('--text_feature_file',
                      dest='text_feature_file',
                      type=str,
                      help='Text Feature file')
    parser.add_argument('--output_tag_transformer_file',
                      dest='dictionary_mapping_file',
                      type=str,
                      help='Tag transformer dictionary file')


    args = parser.parse_args()
    dataset = Dataset(args.image_feature_file, args.text_feature_file)

    print('Creating One hot')
    oh = OneHot(dataset)
    print('OneHot Encoding with {} keys'.format(oh.num_keys))
    oh.save_data_mapping(args.dictionary_mapping_file)


if __name__ == '__main__':
    main()
