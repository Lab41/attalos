from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import gzip
import numpy as np

import h5py


class Dataset(object):
    """
    The Dataset class provides a Python class to represent the output of the prerpocessing stage.
    Any algorithm that uses the dataset class interface should be able to take advantage of any
    dataset for which a attalos.dataset.dataset_prep class has been written.
    """

    TEXT_FEAT_TYPES_AVAILABLE = set(['tags', 'captions'])
    def __init__(self, img_feature_filename, text_feature_filename, text_feat_type="tags", seed=1024):
        self.rng = np.random.RandomState(seed)
        self.img_feature_filename = img_feature_filename
        self.text_feature_filename = text_feature_filename
        self.text_feat_type = text_feat_type
        if self.text_feat_type not in self.TEXT_FEAT_TYPES_AVAILABLE:
            raise NotImplementedError('Feature type {} not yet available. Please choose from: {}'
                                      .format(text_feat_type, ','.join(self.TEXT_FEAT_TYPES_AVAILABLE)))

        self.__load_image_features()
        self.__load_text_features()

    def __load_image_features(self):
        self.img_feature_file = h5py.File(self.img_feature_filename)
        self.image_feats = self.img_feature_file['feats']
        self.image_ids = self.img_feature_file['ids']
        self.img_feat_size = len(self.image_feats[0,:]) # get length of the first feature vector
        self.num_images = len(self.image_ids)

    def __load_text_features(self):
        if self.text_feature_filename.endswith('.gz'):
            input_file = gzip.open(self.text_feature_filename)
        else:
            input_file = open(self.text_feature_filename)
        self.text_feats = json.load(input_file)[self.text_feat_type]

    def get_next_batch(self, batch_size):
        items_in_batch = np.random.randint(0, self.num_images, batch_size)

        img_feats = np.zeros((batch_size, self.img_feat_size))
        text_feats = []
        for i, item_index in enumerate(items_in_batch):
            item_id = self.image_ids[item_index]
            if not isinstance(item_id, str):
                item_id = str(item_id)
            img_feats[i, :] = self.image_feats[item_index, :]
            text_feat = self.text_feats[item_id]
            text_feats.append(text_feat)
        return img_feats, text_feats




def main():
    import argparse
    import time
    parser = argparse.ArgumentParser(description='Exercise Dataset class.')
    parser.add_argument('--image_feature_file',
                      dest='image_feature_file',
                      type=str,
                      help='Image Feature file')
    parser.add_argument('--text_feature_file',
                      dest='text_feature_file',
                      type=str,
                      help='Text Feature file')
    args = parser.parse_args()
    t = Dataset(args.image_feature_file, args.text_feature_file)

    start_time = time.time()
    for i in range(10):
        img_feats, text_feats = t.get_next_batch(128)
    print('Time to extract 10 batches: {}'.format(time.time() - start_time))
    
if __name__ == '__main__':
    main()
