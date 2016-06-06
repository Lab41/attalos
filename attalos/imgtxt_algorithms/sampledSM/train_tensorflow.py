from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

import negsamp

from attalos.dataset.dataset import Dataset
from attalos.dataset.onehot import OneHot

def run_the_net(
        train_dataset,
        test_dataset,
        onehot_encoding,
        output_directory=None,
        output_prefix=None,
        batch_size=128,
        hidden_layer_size=100,
        learning_rate=0.001,
        negative_sample_count=10,
        training_epochs=10
    ):

    # Set the size of our vectors from the data
    image_feature_size = train_dataset.img_feat_size
    vocabulary_size = onehot_encoding.vocab_size

    # Placeholder tensors (feed_dict must be used in the run call to replace these
    # with data)
    VARIABLE_SIZE = None  # None in a shape means "variable size"
    hidden_layer_inputs = tf.placeholder(tf.float32, shape=[VARIABLE_SIZE, image_feature_size], name="Input")
    positive_examples = tf.placeholder(tf.float32, shape=[VARIABLE_SIZE, vocabulary_size], name="Positive_Examples")
    negative_examples = tf.placeholder(tf.float32, shape=[VARIABLE_SIZE, vocabulary_size], name="Negative_Examples")

    # Weights, stddev is adjusted to account for the very large change from 4000+ inputs to 100 outputs
    weights_hidden = tf.Variable(tf.random_normal([image_feature_size, hidden_layer_size], stddev=0.05), name="hidden_weights")
    weights_output = tf.Variable(tf.random_normal([hidden_layer_size, vocabulary_size], stddev=0.3), name="output_weights")

    # The single layer perceptron
    def single_layer_perceptron(hidden_layer_inputs, weights_hidden, weights_output):
        hidden_layer = tf.matmul(hidden_layer_inputs, weights_hidden)
        #hidden_layer = tf.nn.sigmoid(tf.matmul(hidden_layer_inputs, weights_hidden))
        return tf.matmul(hidden_layer, weights_output), hidden_layer

    model_outputs, hidden_layer = single_layer_perceptron(hidden_layer_inputs, weights_hidden, weights_output)

    # The loss function
    poss_times_model = tf.mul(positive_examples, tf.log(tf.sigmoid(model_outputs)))
    neg_times_model =  tf.mul(negative_examples, tf.log(tf.sigmoid(-model_outputs)))

    poss_loss = tf.reduce_mean(poss_times_model)
    neg_loss = tf.reduce_mean(neg_times_model)

    loss = -1. * (poss_loss + neg_loss)

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Set up negative samples
    word_count = np.zeros(onehot_encoding.vocab_size)
    for dataset in (train_dataset, test_dataset):
        for item_index in dataset:
            img, text = dataset[item_index]
            word_count += onehot_encoding.get_multiple(text)

    negative_sample_generator = negsamp.NegativeSampler(word_count)

    init_op = tf.initialize_all_variables()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    if output_directory is not None:
        saver = tf.train.Saver([weights_hidden, weights_output])

    with tf.Session(config=config) as sess:

        # Initialize the tensors
        sess.run(init_op)

        # Training cycle
        for epoch in range(training_epochs):
            print("Training {}".format(epoch))
            for batch in range(int(train_dataset.num_images / batch_size)):
                # Get a batch
                train_image_features, text_tags = train_dataset.get_next_batch(batch_size)
                train_text_features = np.array(map(onehot_encoding.get_multiple, text_tags))

                negative_samples = negative_sample_generator.negsampv(train_text_features, negative_sample_count)

                _, loss_result = sess.run(
                    [optimizer, loss],
                    feed_dict={
                        hidden_layer_inputs: train_image_features,
                        positive_examples: train_text_features,
                        negative_examples: negative_samples,
                    }
                )

            print("\tLoss: {:.5}".format(loss_result))

            # Test model
            test_results = sess.run(
                model_outputs,
                feed_dict={
                    hidden_layer_inputs: train_image_features,
                }
            )
            # Get the accuracy
            model_predicted_indices = np.argpartition(test_results, -5)[:,-5:]
            true_indices = np.argpartition(train_text_features, -5)[:, -5:]

            total = 0
            for i in range(len(model_predicted_indices)):
                total += len(np.intersect1d(model_predicted_indices[i], true_indices[i])) / 5.

            total /= float(len(model_predicted_indices))
            print("\tAccuracy: {:.2%}\n".format(total))

            if output_directory is not None and not epoch % 100:
                # Save the variables to disk.
                output_loc = output_directory + "/" + output_prefix
                save_path = saver.save(sess, output_loc, global_step=epoch)
                print("Model saved in file: {}".format(save_path))

        print("Optimization Finished!")


def main():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument(
        "image_feature_file_train",
        type=str,
        help="Image Feature file for the training set",
    )
    parser.add_argument(
        "text_feature_file_train",
        type=str,
        help="Text Feature file for the training set",
    )
    parser.add_argument(
        "image_feature_file_test",
        type=str,
        help="Image Feature file for the test set",
    )
    parser.add_argument(
        "text_feature_file_test",
        type=str,
        help="Text Feature file for the test set",
    )

    # Optional arguments
    parser.add_argument(
        "--output_directory",
        "-d",
        dest="output_directory",
        type=str,
        help="Directory to save the checkpoint files in",
        default=None,
    )
    parser.add_argument(
        "--output_prefix",
        "-p",
        dest="output_prefix",
        type=str,
        help="Text to prefix the checkpoint files with",
        default="single_layer_perceptron_with_negative_sampling",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        dest="batch_size",
        type=int,
        help="Batch size [default 128]",
        default=128,
    )
    parser.add_argument(
        "--hidden_layer_size",
        "-s",
        dest="hidden_layer_size",
        type=int,
        help="Size of the hidden layer [default 100]",
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        dest="learning_rate",
        type=float,
        help="Learning rate [default 0.001]",
        default=0.001,
    )
    parser.add_argument(
        "--negative_sample_count",
        "-n",
        dest="negative_sample_count",
        type=int,
        help="The number of negative samples to generate per image [default 10]",
        default=10,
    )
    parser.add_argument(
        "--training_epochs",
        "-e",
        dest="training_epochs",
        type=int,
        help="The number of training epochs [default 10]",
        default=10,
    )

    args = parser.parse_args()

    # Read in the data and set up the text encoding
    train_image_feature_file = args.image_feature_file_train
    train_text_feature_file = args.text_feature_file_train
    train_dataset = Dataset(train_image_feature_file, train_text_feature_file)

    test_image_feature_file = args.image_feature_file_test
    test_text_feature_file = args.text_feature_file_test
    test_dataset = Dataset(test_image_feature_file, test_text_feature_file)

    onehot_encoding = OneHot([train_dataset, test_dataset])

    # Train the network
    run_the_net(
        train_dataset,
        test_dataset,
        onehot_encoding,
        args.output_directory,
        args.output_prefix,
        args.batch_size,
        args.hidden_layer_size,
        args.learning_rate,
        args.negative_sample_count,
        args.training_epochs
    )


if __name__ == "__main__":
    main()
