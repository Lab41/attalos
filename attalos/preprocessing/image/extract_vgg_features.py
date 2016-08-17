from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import subprocess
import re

import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import h5py


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with open('vgg16-20160129.tfmodel', mode='rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def load_image(path):
  # load image
  img = imread(path)
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = imresize(crop_img, (224, 224))
  return resized_img

def run_inference_on_dataset(dataset, tmp_dir='/tmp/'):
  """Runs inference on an image.
  Args:
    dataset (DatasetPrep): Dataset
    tmp_dir (str): Directory to store images temporarily
  Returns:
    Nothing
  """
  # Creates graph from saved GraphDef.
  create_graph()
  image_keys = dataset.list_keys()
  features = np.zeros((len(image_keys), 4096), dtype=np.float16)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    for ind, img_record in enumerate(dataset):
        if ind % 1000 == 0:
            print ('Completed %d of %d'%(ind, len(image_keys)))

        new_fname = os.path.join(tmp_dir, os.path.basename(img_record.image_name))
        dataset.extract_image_to_location(img_record.id, new_fname)

        try:
            image_data = load_image(new_fname)
            image_data = np.array(image)[:, :, 0:3]  # Select RGB channels only.

        except: # Not a jpeg, use file to find extension, try to read with scipy
            filetype = subprocess.Popen(["file", new_fname], stdout=subprocess.PIPE).stdout.read()
            extension = re.search(r':[ ]+([A-Z]+) ', filetype).group(1).lower()
            new_new_fname = new_fname + '.{}'.format(extension)
            print('Renaming to {}'.format(new_new_fname))
            shutil.move(new_fname, new_new_fname)
            image = load_image(new_new_fname) #Image.open(new_fname)
            image_data = np.array(image)[:, :, 0:3]  # Select RGB channels only.
        try:
            image_data = image_data.reshape((1, 224, 224, 3))
            pool_3_tensor = sess.graph.get_tensor_by_name('fc7/BiasAdd:0')
            predictions = sess.run(pool_3_tensor,
                                   {'images:0': image_data})
        except:
            print('Error on {}: Found dimensions {}'.format(img_record.image_name, image_data.shape))
            raise

        features[ind, :] = np.squeeze(predictions)
        if os.path.exists(new_fname):
            os.remove(new_fname)

    return features

def save_hdf5(local_working_dir, hdf5_fname, image_features, image_ids):
    '''
    Create hdf5 file from features and filename list
    '''
    bname = os.path.basename(hdf5_fname)
    temp_fname = os.path.join(local_working_dir, bname)


    fOut = h5py.File(temp_fname, 'w')
    fOut.create_dataset('ids', data=image_ids)
    fOut.create_dataset('feats', data=image_features, dtype=np.float32)
    fOut.close()

    shutil.move(temp_fname, hdf5_fname)


def process_dataset(dataset_prep, output_fname, working_dir=tempfile.gettempdir()):
  """

  Args:
      dataset_prep (attalos.dataset.DatasetPrep): Dataset to convert
      output_fname: Output filename to extract to
      working_dir: Working directory to use for intermediate files

  Returns:

  """
  # Extract image features using inception network
  # TODO: Maybe this should batch in some way for large jobs?
  features = run_inference_on_dataset(dataset_prep)

  # Save computed features to file
  image_ids = [str(record.id) for record in dataset_prep]
  save_hdf5(working_dir, output_fname, features, image_ids)


def main():
  import argparse

  parser = argparse.ArgumentParser(description='Extract image features using VGG model.')
  parser.add_argument('--dataset_dir',
                      dest='dataset_dir',
                      type=str,
                      help='Directory with input images')
  parser.add_argument('--dataset_type',
                      dest='dataset_type',
                      default='mscoco',
                      choices=['mscoco', 'visualgenome', 'iaprtc', 'generic', 'espgame', 'nuswide'])
  parser.add_argument('--split',
                      dest='split',
                      default='train',
                      choices=['train', 'test', 'val'])
  parser.add_argument('--output_fname',
                      dest='output_fname',
                      default='image_features.hdf5',
                      type=str,
                      help='Output hd5f filename')
  parser.add_argument('--working_dir',
                      dest='working_dir',
                      default=tempfile.gettempdir(),
                      type=str,
                      help='Working directory for hdf5 file creation')
  args = parser.parse_args()

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
  elif args.dataset_type == 'generic':
    from attalos.dataset.generic_prep import GenericDatasetPrep
    dataset_prep = GenericDatasetPrep(args.dataset_dir, split=args.split)
  elif args.dataset_type == 'espgame':
    print('Processing espgame data')
    from attalos.dataset.espgame_prep import ESPGameDatasetPrep
    dataset_prep = ESPGameDatasetPrep(args.dataset_dir, split=args.split)
  elif args.dataset_type == 'nuswide':
    print('Processing nuswide data')
    from attalos.dataset.nuswide_prep import NUSWideDatasetPrep
    dataset_prep = NUSWideDatasetPrep(args.dataset_dir, split=args.split)
  else:
      raise NotImplementedError('Dataset type {} not supported'.format(args.dataset_type))
  process_dataset(dataset_prep, args.output_fname, working_dir=args.working_dir)


if __name__ == '__main__':
  main()

