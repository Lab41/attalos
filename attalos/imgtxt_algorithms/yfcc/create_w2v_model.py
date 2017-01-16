import gensim
import glob
import bz2
import six
import datetime
import os
import argparse

class Sentences(object):
    def __init__(self, filenames):
        self.filenames = filenames
 
    def __iter__(self):
      for filename in self.filenames:
        print(filename, datetime.datetime.now())
        for line in bz2.BZ2File(filename):
            tags = line.decode('ascii').split('\t')[8].strip()
            if len(tags) > 0:
                yield [six.moves.urllib.parse.unquote(tag)  for tag in tags.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Created word2vec model for yfcc data')
    parser.add_argument("input_directory",
                        type=str,
                        help="Directory containing yfcc100m_dataset-[0-9].bz2")
    parser.add_argument("output_filename",
                        type=str,
                        help="Output w2v model name")

    args = parser.parse_args()
    filenames = sorted(glob.glob(os.path.join(args.input_directory, 'yfcc100m_dataset-*.bz2')))
    w2v_model = gensim.models.Word2Vec(Sentences(filenames), size=200, window=5, min_count=50, workers=20)

    w2v_model.save_word2vec_format(args.output_filename)


if __name__ == '__main__':
    main()
