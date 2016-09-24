import math
import numpy as np
import h5py
import tensorflow as tf

# Attalos Imports
import attalos.util.log.log as l
from attalos.dataset.dataset import Dataset
from attalos.imgtxt_algorithms.main import load_wv_model, ModelTypes, WordVectorTypes

logger = l.getLogger(__name__)


def infer_dataset(args):
    logger.info("Parsing train and test datasets.")
    dataset = Dataset(args.image_feature_file_train, args.text_feature_file_train)

    logger.info("Reading word vectors from file.")
    wv_model = load_wv_model(args.word_vector_file, args.word_vector_type)

    output_file = h5py.File(args.output_hdf5_file, 'w')
    output_file.create_dataset('preds', (dataset.num_images, wv_model.get_word_vector_shape()[0]),
                               dtype=np.float32)

    with tf.Graph().as_default():
        model_cls = ModelTypes[args.model_type].value
        logger.info("Selecting model class: %s" % model_cls.__name__)
        datasets = [dataset]
        model = model_cls(wv_model, datasets, **vars(args))

        logger.info("Preparing test_dataset.")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        batch_size = args.batch_size
        image_ids = []
        with tf.Session(config=config) as sess:
            model.initialize_model(sess)
            if args.model_input_path:
                logger.info("Loading stored weights")
                model.load(sess, args.model_input_path)
            else:
                logger.error('Inference requires a saved model to be loaded')
                raise ValueError('Inference requires a saved model')

            num_batches = math.ceil(dataset.num_images/batch_size)
            for batch_num in range(int(num_batches)):
                logger.info('Batch {} of {}'.format(batch_num, num_batches))
                start_index = batch_num*batch_size
                stop_index = min((batch_num+1)*batch_size, dataset.num_images)
                image_features = dataset.image_feats[start_index:stop_index]
                output_feats = model.predict_feats(sess, image_features)
                print(output_feats.shape, output_file['preds'][start_index:stop_index].shape)
                output_file['preds'][start_index:stop_index] = output_feats

        output_file.create_dataset('ids', data=image_ids)
        output_file.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Two layer linear regression')
    parser.add_argument("image_feature_file_train",
                        type=str,
                        help="Image Feature file for the training set")
    parser.add_argument("text_feature_file_train",
                        type=str,
                        help="Text Feature file for the training set")
    parser.add_argument("output_hdf5_file",
                        type=str,
                        help="Output HDF5 file")
    parser.add_argument("word_vector_file",
                        type=str,
                        help="Text file containing the word vectors")

    # Optional Args
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning Rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size to use for training")
    parser.add_argument("--model_type",
                        type=str,
                        default="multihot",
                        choices=['multihot', 'naivesum', 'wdv', 'negsampling', 'fast0tag'],
                        help="Loss function to use for training")
    parser.add_argument("--model_input_path",
                        type=str,
                        default=None,
                        help="Model input path (to continue training)")

    # new args
    parser.add_argument("--hidden_units",
                        type=str,
                        default="200,200",
                        help="Define a neural network as comma separated layer sizes")
    parser.add_argument("--word_vector_type",
                        type=str,
                        choices=[t.name for t in WordVectorTypes],
                        help="Format of word_vector_file")
    parser.add_argument("--optim_words",
                        action="store_true",
                        default=False,
                        help="If using negsampling model_type, use to jointly optimize words")
    parser.add_argument("--joint_factor",
                        type=float,
                        default=1.0,
                        help="Multiplier for learning rate in updating joint optimization")

    args = parser.parse_args()
    infer_dataset(args)


if __name__ == '__main__':
    main()
