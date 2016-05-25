# Borrows extensively from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile
import shutil

import numpy as np
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

def run_inference_on_image(image_list):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  # Creates graph from saved GraphDef.
  create_graph()
  features = np.zeros((len(image_list), 2048), dtype=np.float16)
  with tf.Session() as sess:
    for ind, image in enumerate(image_list):
        if ind % 1000 == 0:
            print ('Complted %d of %d'%(ind, len(image_list)))
        if not tf.gfile.Exists(image):
            tf.logging.fatal('File does not exist %s', image)
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        pool_3_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        predictions = sess.run(pool_3_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        features[ind, :] = np.squeeze(predictions)
    return features

def save_hdf5(local_working_dir, hdf5_fname, image_features, im_files_for_batch):
    '''
    Create hdf5 file from features and filename list
    '''
    bname = os.path.basename(hdf5_fname)
    temp_fname = os.path.join(local_working_dir, bname)

    im_files_wo_fnames = [os.path.basename(file) for file in im_files_for_batch]
    fOut = h5py.File(temp_fname, 'w')
    fOut.create_dataset('filenames', data=im_files_wo_fnames)
    fOut.create_dataset('feats', data=image_features, dtype=np.float32, compression='gzip')
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

def main(_):
  import glob
  maybe_download_and_extract()
  image_filename_list = glob.glob('/local_data/train2014/*')[:100]
  image_names_only_filename_list = [os.path.basename(fname) for fname in image_filename_list]
  features = run_inference_on_image(image_filename_list)
  save_hdf5('/tmp', '/local_data/feats.hdf5', features, image_names_only_filename_list)


if __name__ == '__main__':
  tf.app.run()

