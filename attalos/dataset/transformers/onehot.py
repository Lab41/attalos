from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from attalos.dataset.transformers.texttransformer import TextTransformer

class OneHot(TextTransformer):
    """
    Transforms tags from a dataset iterator into into a one-hot encoding
    """
    def __init__(self, datasets, valid_vocab=None, dictionary_file=None):
        """
        Initialize OneHot encoding
        Args:
            dataset (attalos.dataset.dataset): A  dataset iterator (or list of iterators)
            dictionary_file: A saved dictionary file

        Returns:

        """
        super(OneHot, self).__init__(dictionary_file)
        if dictionary_file:
            self.vocab_size = len(super(OneHot, self).keys())
        else:
            self.data_mapping = {}
            self.create_data_mapping(datasets, valid_vocab)

    def create_data_mapping(self, *args, **kwargs):
        datasets = args[0]
        valid_vocab = args[1]
        dataset_tags = set()
        if isinstance(datasets, collections.Iterable):
            iterable_datasets = datasets
        else:
            iterable_datasets = [datasets]

        for dataset in iterable_datasets:
            for tags in dataset.text_feats.values():
                dataset_tags.update(tags)
        
        if valid_vocab:
            dataset_tags = filter(lambda x: x in valid_vocab, dataset_tags)
            
        self.vocab_size = len(dataset_tags)
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
        multihot_feats = np.zeros(self.vocab_size)
        for tag in tags:
            if tag in self.data_mapping:
                multihot_feats += self.__getitem__(tag)
        return multihot_feats

    def __getitem__(self, item):
        if item not in self.data_mapping:
            return None
        index = self.data_mapping[item]
        arr = np.zeros(self.vocab_size)
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
    print('OneHot Encoding with {} keys'.format(oh.vocab_size))
    oh.save_data_mapping(args.dictionary_mapping_file)


if __name__ == '__main__':
    main()
