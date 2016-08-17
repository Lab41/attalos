# Borrows extensively from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import shutil
import tempfile
import subprocess
import re

import numpy as np
from scipy.misc import imread
from six.moves import urllib
import tensorflow as tf
import h5py

FLAGS = tf.app.flags.FLAGS
# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

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
  features = np.zeros((len(image_keys), 2048), dtype=np.float16)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    for ind, img_record in enumerate(dataset):
        if ind % 1000 == 0:
            print ('Completed %d of %d'%(ind, len(image_keys)))

        new_fname = os.path.join(tmp_dir, os.path.basename(img_record.image_name))
        dataset.extract_image_to_location(img_record.id, new_fname)

        try:
            if not tf.gfile.Exists(new_fname):
                tf.logging.fatal('File does not exist %s', new_fname)
            image_data = tf.gfile.FastGFile(new_fname, 'rb').read()

            pool_3_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            predictions = sess.run(pool_3_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
        except: # Not a jpeg, use file to find extension, try to read with scipy
            try:
                filetype = subprocess.Popen(["file", new_fname], stdout=subprocess.PIPE).stdout.read()
                extension = re.search(r':[ ]+([A-Z]+) ', filetype).group(1).lower()
                new_new_fname = new_fname + '.{}'.format(extension)
                print('Renaming to {}'.format(new_new_fname))
                shutil.move(new_fname, new_new_fname)
                image = imread(new_new_fname) #Image.open(new_fname)
                image_data = np.array(image)[:, :, 0:3]  # Select RGB channels only.
                pool_3_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                predictions = sess.run(pool_3_tensor,
                                       {'DecodeJpeg:0': image_data})
            except:
                filetype = subprocess.Popen(["file", new_new_fname], stdout=subprocess.PIPE).stdout.read()
                print('Expected PNG/JPEG, received:  {}'.format(filetype))
                print('Image data size: {}'.format(np.array(image).size()))
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


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def process_dataset(dataset_prep, output_fname, working_dir=tempfile.gettempdir()):
  """

  Args:
      dataset_prep (attalos.dataset.DatasetPrep): Dataset to convert
      output_fname: Output filename to extract to
      working_dir: Working directory to use for intermediate files

  Returns:

  """
  # Download Inception weights if not already present and extract for use
  maybe_download_and_extract()

  # Extract image features using inception network
  # TODO: Maybe this should batch in some way for large jobs?
  features = run_inference_on_dataset(dataset_prep)

  # Save computed features to file
  image_ids = [str(record.id) for record in dataset_prep]
  save_hdf5(working_dir, output_fname, features, image_ids)


def main(_):
  import argparse


  parser = argparse.ArgumentParser(description='Extract image features using Inception model.')
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
    print('Processing espgame data')
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
  tf.app.run()

