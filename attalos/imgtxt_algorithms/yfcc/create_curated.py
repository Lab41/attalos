import h5py
import numpy as np


def get_valid_indices(dataset, w2v_model, max_tags=4):
    num_items = dataset.num_images

    valid_inds = []
    for i in range(num_items):
        item_id = dataset.image_ids[i]
        tags = dataset.text_feats[item_id]
        if 0 < len(tags) < max_tags:
            good_tags = [tag.lower() for tag in tags if tag.lower() in w2v_model]

            if len(good_tags) > 0:
                valid_inds.append(i)
    return valid_inds

def tags_2_vec(tags, w2v_model=None):
    if len(tags) == 0:
        raise ValueError('No Tags')

    word_vectors = []
    good_tags = [tag.replace(' ', '-') for tag in tags if tag.replace(' ', '-') in w2v_model]
    if len(good_tags) == 0:
        raise KeyError('Tags not found: {}'.format(tags))
        return np.zeros(300)

    for tag in good_tags:
        word_vector = w2v_model[tag]
        word_vectors.append(word_vector)
    output = np.sum(word_vectors, axis=0)
    return output /np.linalg.norm(output)

def create_curated_hd5f(input_dataset, output_hdf5_name, word_vectors, max_tags=4):
    # Get valid indices
    valid_indices = get_valid_indices(input_dataset, word_vectors, max_tags=max_tags)

    # Write output hdf5
    image_feats, tags = input_dataset.get_index(valid_indices[0])
    image_feat_size = image_feats.squeeze().shape[0]
    word_feat_size = None
    for tag in tags:
        if tag in word_vectors:
            word_feat_size = word_vectors[tag].squeeze().shape[0]
    fOut = h5py.File(output_hdf5_name, 'w')
    fOut.create_dataset('img_feats', (len(valid_indices), image_feat_size), dtype=np.float16)
    fOut.create_dataset('word_feats', (len(valid_indices), word_feat_size), dtype=np.float16)
    import time
    output_ids = []
    for i, index in enumerate(valid_indices):
        if i%100000 == 0:
            print '{} of {} at {}'.format(i, len(valid_indices), time.time())
        image_feats, tags = input_dataset.get_index(index)
        image_feats /= np.linalg.norm(image_feats)
        word_feats = tags_2_vec(tags, word_vectors)
        fOut['img_feats'][i, :] = image_feats
        fOut['word_feats'][i, :] = word_feats
        output_ids.append( input_dataset.image_ids[index])
    fOut.create_dataset('ids', data=output_ids)
    fOut.close()

def main():
    import gensim
    from attalos.dataset.dataset import Dataset
    import argparse
    parser = argparse.ArgumentParser(description='Created curated hdf5 file to be used by negsampling_regression.py')
    parser.add_argument("image_feature_file",
                        type=str,
                        help="Dataset Image Feature file")
    parser.add_argument("text_feature_file",
                        type=str,
                        help="Dataset Text Feature file")
    parser.add_argument("output_hdf5_filename",
                        type=str,
                        help="Output HDF5 Filename")
    parser.add_argument("word_vector_filename",
                        type=str,
                        help="Gensim word vector file")

    # Optional Arguments
    parser.add_argument("--max_tags",
                        type=int,
                        default=4,
                        help="Max # of tags/image")

    args = parser.parse_args()

    input_dataset = Dataset(args.image_feature_file, args.text_feature_file)
    word_vectors = gensim.models.Word2Vec.load(args.word_vector_filename)

    create_curated_hd5f(input_dataset, args.output_hdf5_filename, word_vectors, max_tags=args.max_tags)


if __name__ == '__main__':
    main()
