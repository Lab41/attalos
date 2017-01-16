import urllib
import json
import gzip
import bz2
import os
import time
I_ID = 0
I_LAT = 11
I_LON = 10
I_TAGS = 8
I_URL = 14


def create_text_json(yfcc_metadata_filename, yfcc_hash_file):
    start_time = time.time()
    # Need to map IDs to Hashes
    id_2_hash = {}
    for line in bz2.BZ2File(yfcc_hash_file):
        ls = line.rstrip().split('\t')
        id_2_hash[ls[0]] = ls[1]
    print('Completed Loading hash', time.time() - start_time)

    print('Starting to process files')
    start_time = time.time()

    output_fname = yfcc_metadata_filename + '.json.gz'
    if os.path.exists(output_fname):
        raise ValueError('File %s already exists'%output_fname)

    fIn = bz2.BZ2File(yfcc_metadata_filename)
    metadata_dict = {}
    for line in fIn:
        ls = line.strip().split('\t')
        hash_val = id_2_hash[ls[I_ID]]
        tags = urllib.unquote(ls[I_TAGS]).decode('utf8').split(',')
        metadata_dict[hash_val] = tags

    output_object  = {'tags':metadata_dict}

    output_file = gzip.open(output_fname, 'w')
    json.dump(output_object, output_file)
    fIn.close()
    output_file.close()

    print('Elapsed Time: {}'.format(time.time() - start_time))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create text json normally created by preprocessing.text')
    parser.add_argument("yfcc_hash_file",
                        type=str,
                        help="/path/to/yfcc_100m_hash.bz2")

    parser.add_argument("yfcc_metadata_filename",
                        type=str,
                        help="/path/to/yfcc100m_dataset-X.bz2")

    args = parser.parse_args()
    create_text_json(args.yfcc_metadata_filename, args.yfcc_hash_file)


if __name__ == '__main__':
    main()
