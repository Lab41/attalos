import h5py
import tarfile
import numpy as np
import time
import re
import argparse

def process_file(filename):
    tfile = tarfile.open(filename, "r")
    fOut = h5py.File(filename + '.hdf5', 'w')
    for tarinfo in tfile:
        print tarinfo.name, "is", tarinfo.size, "bytes in size and is",
        if tarinfo.isreg():
            batch_num = int(re.search('batch_([0-9]+).txt', tarinfo.name).group(1))
            image_ids = []
            image_features = []
            start_time = time.time()
            f = tfile.extractfile(tarinfo.name)
            print 'starting: {}'.format(tarinfo.name)
            line_count = 0
            for line in f:
                ls = line.strip().split('\t', 3)
                image_id = ls[0]
                net = ls[1]
                dimension = ls[2]

                image_feature = np.fromstring(ls[3], dtype=np.float16, sep=' ')
                image_ids.append(image_id)
                image_features.append(image_feature)
                line_count +=1
            print 'Extracted {} lines in {}'.format(line_count, time.time() - start_time)
            fOut.create_dataset('ids_%d'%batch_num, data=image_ids)
            fOut.create_dataset('feats_%d'%batch_num, data=np.array(image_features), dtype=np.float16)
        print 'Completed loop'

    tfile.close()
    fOut.close()

def main():
    parser = argparse.ArgumentParser(description='Create VGG feature HDF5 file to mimic image preprocessing')
    parser.add_argument("image_feature_file",
                        type=str,
                        help="Dataset Image Feature file")


    args = parser.parse_args()
    process_file(args.image_feature_file)
if __name__ == '__main__':
    main()
