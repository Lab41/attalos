import numpy as np
from attalos.dataset.dataset import Dataset
from attalos.dataset.transformers.onehot import OneHot


def load_entire_dataset(dataset, split='test'):

    if dataset=='yfcc':
        image_feature_file = '/local_data/yonas/yfcc_vgg/yfcc100m_dataset-0.modified3.valid_only.hdf5'
        text_feature_file = '/local_data/yonas/yfcc_metadata/yfcc_text.json.gz'
        # all_train_dataset = Dataset(image_feature_file, text_feature_file)
        imdir = '/data/fs4/datasets/iaprtc-12/images/images/'
    elif dataset=='iaprtc12':
        imdata = np.load('/data/fs4/datasets/iaprtc-12/iaprtc12-inria.npz')
        imdir = '/data/fs4/datasets/iaprtc-12/images/images/'
        if split=='train':
            x = imdata['xTr']
            y = imdata['yTr']
        else:
            x = imdata['xTe']
            y = imdata['yTe']
        imD = imdata['D']
        trainlist = imdata['trainlist']
        testlist = imdata['testlist']
        imdata.files
    elif dataset=='espgame':
        imdata = np.load('/data/fs4/datasets/espgame/espgame-inria.npz')
        imdir = '/data/fs4/datasets/espgame/ESP-ImageSet/images/'
        if split=='train':
            x = imdata['xTr']
            y = imdata['yTr']
        else:
            x = imdata['xTe']
            y = imdata['yTe']
        imD = imdata['D']
        trainlist = imdata['trainlist']
        testlist = imdata['testlist']
        imdata.files
    return x, y, imD, trainlist, testlist, imdir


def load_entire_dataset_di(dataset, datadir='/data/fs4/teams/attalos/features/', split='test', allhot=None):

    if dataset == 'iaprtc12':
        if split=='train':
            imdata=datadir+'image/iaprtc_train_20160816_inception.hdf5'
            txdata=datadir+'text/iaprtc_train_20160816_text.json.gz'
        else:
            imdata=datadir+'image/iaprtc_test_20160816_inception.hdf5'
            txdata=datadir+'text/iaprtc_test_20160816_text.json.gz'
        imdir = '/data/fs4/datasets/iaprtc-12/images/images/'
    if dataset == 'espgame':
        if split=='train':
            imdata=datadir+'image/espgame_train_20160816_inception.hdf5'
            txdata=datadir+'text/espgame_train_20160816_text.json.gz'
        else:
            imdata=datadir+'image/espgame_test_20160823_inception.hdf5'
            txdata=datadir+'text/espgame_test_20160823_text.json.gz'
        imdir = '/data/fs4/datasets/espgame/ESP-ImageSet/images/'
    
    # Training data
    data = Dataset(imdata, txdata)

    # Training image features
    x = np.array( data.image_feats )

    # Image lists (in the order of the images in the dataset)
    imlist = [imid for imid in data.image_ids]

    # One hot encoding
    if not allhot:
        allhot = OneHot([data])
    y = np.array([ allhot.get_multiple( data.text_feats[imid] ) for imid in data.image_ids ])

    # Dictionaries of the one hot encoding
    dTr = [imid for imid in allhot.data_mapping]

    return x, y, dTr, imlist, imdir, allhot
