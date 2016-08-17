from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import gzip


def extract_text_features(dataset_prep):
    """
    Extract captions and tags from dataset_prep object
    Args:
        dataset_prep: (attalos.dataset.DatasetPrep): Dataset to extract features from

    Returns:

    """
    tag_dict = {}
    caption_dict = {}
    for img_record in dataset_prep: # img_record (attalos.dataset.dataset_prep.RecordMetadata)
        id = str(img_record.id) # Keys should be strings
        tags = img_record.tags
        captions = img_record.captions
        tag_dict[id] = tags
        caption_dict[id] = captions
    return tag_dict, caption_dict


def process_dataset(dataset_prep, output_fname):
    """
    Uses dataset_prep object to extract text features

    Args:
      dataset_prep (attalos.dataset.DatasetPrep): Dataset to convert
      output_fname: Output filename to extract to

    Returns:

    """

    tag_dict, caption_dict = extract_text_features(dataset_prep)

    output_object = {'tags': tag_dict, 'captions': caption_dict}
    if output_fname.endswith('.gz'):
        output_file = gzip.open(output_fname, 'w')
    else:
        output_file = open(output_fname, 'w')
    json.dump(output_object, output_file)


def main():
    import argparse
    from attalos.dataset.mscoco_prep import MSCOCODatasetPrep

    parser = argparse.ArgumentParser(description='Extract text features.')
    parser.add_argument('--dataset_dir',
                      dest='dataset_dir',
                      type=str,
                      help='Directory with input data')
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
                      default='captions_text.json.gz',
                      type=str,
                      help='Output json filename')

    args = parser.parse_args()
    if args.dataset_type == 'mscoco':
        from attalos.dataset.mscoco_prep import MSCOCODatasetPrep
        print('Processing MSCOCO Data')
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
        print('Processing generic dataset')
        from attalos.dataset.generic_prep import GenericDatasetPrep
        dataset_prep = GenericDatasetPrep(args.dataset_dir, split=args.split)
    elif args.dataset_type == 'espgame':
        print('Processing espgame')
        from attalos.dataset.espgame_prep import ESPGameDatasetPrep
        dataset_prep = ESPGameDatasetPrep(args.dataset_dir, split=args.split)
    elif args.dataset_type == 'nuswide':
        print('Processing nuswide data')
        from attalos.dataset.nuswide_prep import NUSWideDatasetPrep
        dataset_prep = NUSWideDatasetPrep(args.dataset_dir, split=args.split)
    else:
        raise NotImplementedError('Dataset type {} not supported'.format(args.dataset_type))

    process_dataset(dataset_prep, args.output_fname)

if __name__ == '__main__':
    main()
