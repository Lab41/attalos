
from __future__ import absolute_import
from __future__ import print_function

import os
import tempfile
import numpy as np
import tensorflow as tf
from zipfile import ZipFile
from attalos.preprocessing.image.vgg16 import VGG16
from attalos.preprocessing.image.image_producer import ZipImageProducer


def display_results(image_paths, softmax, imagenet="weights/imagenet-classes.txt"):
    """
    A method for checking that the network outputs make sense
    Args:
        image_paths: list of jpg files
        softmax: softmax output of a model
        imagenet: file that maps the output to the correct label

    Returns: None
    """
    with open(imagenet) as infile:
        class_labels = map(lambda x: x.strip(), infile.readlines())
    class_indices = np.argmax(softmax, axis=1)
    for img_idx, image_path in enumerate(image_paths):
        image_path = image_paths[img_idx]
        img_name = os.path.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = softmax[img_idx, class_indices[img_idx]]
        print('Image {0} classified as {1} ({2:0.2%} confidence)'.format(img_name, class_name, confidence))


def filter_files(archive, ext="jpg"):
    with ZipFile(archive, 'r') as archive:
        return list(filter(lambda x: x.endswith(ext), archive.namelist()))


def run_inference(archive, model_dir, **kwargs):
    # TODO Return some feature output
    # If there is no tensorflow checkpoint file, the program needs these to set
    # the initial network weights
    pretrained_weights_npy = kwargs.pop("npy_weights", "weights/vgg16_weights.npy")

    image_paths = filter_files(archive)

    graph = tf.Graph()
    with graph.as_default():
        data = ZipImageProducer(archive, image_paths)
        model = VGG16(1000)
        initop = tf.initialize_all_variables()
        train_vars_op = tf.trainable_variables()
        saver = tf.train.Saver(var_list=train_vars_op)

    sess = tf.Session(graph=graph)
    # features = np.zeros(shape=(len(dataset.list_keys()), 2048), dtype=np.float32)
    with sess.as_default():
        try:
            sess.run([initop])

            checkpoint = tf.train.get_checkpoint_state(model_dir)
            if checkpoint is not None:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Weights restored!")
            else:
                pretrained_features = np.load(pretrained_weights_npy).item()

                # This will map the numpy VGG16 weights to the tensorflow ops
                # This mapping is weak and should use something more robust
                # like a tensorflow graph or checkpoint object
                for op in train_vars_op:
                    #Tensorflow ops are of the form "scope/varname:0"
                    op_name = op.name.split(":")[0]
                    layer, var = op_name.split("/")
                    sess.run(op.assign(pretrained_features[layer][var]))

            coord = tf.train.Coordinator()
            threads = data.start(sess, coord)
            while True:
                try:
                    _, input_images = data.get_batch(sess)

                    feed = {
                        model.input_data: input_images
                    }

                    features = sess.run(model.softmax, feed_dict=feed)

                    print(features.mean())

                except tf.errors.OutOfRangeError:
                    # This is a "feature" of Queues. When they're finished they throw this exception
                    # so we'll catch it and mark the process complete.
                    break

        except Exception as e:
            raise e

        finally:
            print("Cleaning up...")
            saver.save(sess, os.path.join(model_dir, "vgg16.model"))
            tf.train.write_graph(sess.graph_def, model_dir, "vgg16.graph", as_text=False)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=2)

            print("Complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract image features using VGG16")
    parser.add_argument('--dataset_dir',
                        dest='dataset_dir',
                        type=str,
                        help='Directory with input images')
    parser.add_argument('--dataset_type',
                        dest='dataset_type',
                        default='mscoco',
                        choices=['mscoco', 'visualgenome', 'iaprtc12'])
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
    elif args.dataset_type == 'iaprtc12':
        print('Processing IAPRTC-12 data')
        from attalos.dataset.iaprtc12_prep import IAPRTC12DatasetPrep

        dataset_prep = IAPRTC12DatasetPrep(args.dataset_dir, split=args.split)
    else:
        raise NotImplementedError('Dataset type {} not supported'.format(args.dataset_type))

    process_dataset(dataset_prep, args.output_fname, working_dir=args.working_dir)

if __name__ == "__main__":
    archive = "/Users/andreww/Desktop/sample_data.zip"
    model = "/Users/andreww/Documents/attalos/attalos/preprocessing/image/weights"
    run_inference(archive, model)