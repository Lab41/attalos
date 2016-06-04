from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from texttransformer import TextTransformer

class OneHot(TextTransformer):

    def __init__(self, dictionary_file=None):

        super(OneHot, self).__init__(dictionary_file)
        if dictionary_file:
            self.num_keys = len(super(OneHot, self).keys())

    def create_data_mapping(self, data_prep):
        keys = set()
        count = 0
        total_tags = 0
        for record in data_prep:
            count += 1
            for tag in record.tags:
                keys.add(tag)
            total_tags += len(record.tags)
        print('Count: {}'.format(count))
        print('Tags: {}'.format(total_tags))
        print('Length: {}'.format(len(keys)))
        self.num_keys = len(keys)
        self.data_mapping = {}
        for i, key in enumerate(keys):
            self.data_mapping[key] = i
        print(self.data_mapping)

    def __getitem__(self, item):
        index = self.data_mapping[item]
        arr = np.zeros(self.num_keys)
        arr[index] = 1
        return arr


def main():
    import argparse


    parser = argparse.ArgumentParser(description='Extract image features using Inception model.')
    parser.add_argument('--dataset_dir',
                      dest='dataset_dir',
                      type=str,
                      help='Directory with input images')
    parser.add_argument('--dataset_type',
                      dest='dataset_type',
                      default='mscoco',
                      choices=['mscoco', 'visualgenome', 'iaprtc'])
    parser.add_argument('--split',
                      dest='split',
                      default='train',
                      choices=['train', 'test', 'val'])
    parser.add_argument('--dictionary_mapping_file',
                      dest='dictionary_mapping_file',
                      type=str,
                      help='Text dictionary_mapping_file file')
    args = parser.parse_args()
    print('Loading Datset Prep')
    if args.dataset_type == 'mscoco':
        print('Processing MSCOCO Data')
        from attalos.dataset.mscoco_prep import MSCOCODatasetPrep
        dataset_prep = MSCOCODatasetPrep(args.dataset_dir, split=args.split)
    elif args.dataset_type == 'visualgenome':
        print('Processing Visual Genome Data')
        from attalos.dataset.vg_prep import VGDatasetPrep
        dataset_prep = VGDatasetPrep(args.dataset_dir, split=args.split)
    elif args.dataset_type == 'iaprtc':
        print('Processing IAPRTC-12 data')
        from attalos.dataset.iaprtc12_prep import IAPRTC12DatasetPrep
        dataset_prep = IAPRTC12DatasetPrep(args.dataset_dir, split=args.split)
    else:
        raise NotImplementedError('Dataset type {} not supported'.format(args.dataset_type))

    print('Creating One hot')
    oh = OneHot()
    oh.create_data_mapping(dataset_prep)

    key = dataset_prep.list_keys()[5]
    record = dataset_prep.get_key(key)
    print(record.tags[0])
    print(oh[record.tags[0]].size)
    print(np.sum(oh[record.tags[0]]))
    oh.save_data_mapping(args.dictionary_mapping_file)

if __name__ == '__main__':
    main()
