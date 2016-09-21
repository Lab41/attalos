import sys
sys.path.append('/home/kni/local-kni/attalos')
from attalos.dataset.dataset import Dataset
# from attalos.dataset.transformers.onehot import OneHot
import numpy as np
import json


datadir='/data/fs4/teams/attalos/features/'
imdata=datadir+'image/visualgenome_train_20160816_inception.hdf5'
txdata=datadir+'text/visualgenome_train_20160816_text.json.gz'
dirTr = '/data/fs4/datasets/vg_unzipped/'
splits = json.load(open('densecap_splits.json'))                                                                                    

## Using our dataset iterators, load in the text data
alldata = Dataset(imdata,txdata)
for key in alldata.text_feats:
    words = alldata.text_feats[key]
    words = [ word.split()[-1].lower() for word in words ]
    alldata.text_feats[key] = words
del key,words,word


## Create an unordered vocabulary with counts
#  Full vocabulary is stored in `vocab`, with `counts`
def getvocab( text_feats ):
    
    vocab = dict()
    for key in text_feats:
        words = text_feats[key]
        for word in set(words):
            if vocab.has_key(word):
                vocab[word]+=1
            else:
                vocab[word]=1
                
    return vocab


# ## Order the vocabulary and counts
# 
# - The full dictionary is stored in `D`
# - the full counts are stored in `C`
def ordervocab(vocab):
    
    listcount = []
    listwords = []
    for i,word in enumerate(vocab):
        listcount += [vocab[word]]
        listwords += [word]
    indices = np.argsort( np.array(listcount) )[::-1]
    returnD = [listwords[i] for i in indices]
    returnC = [listcount[i] for i in indices]
    
    return returnD, returnC


# ## Cutoffs
# - New dictionary for training is `dTr`
# - New counts for training is `cTr`
# - The same information is stored in `dTrHash`, only as a dictionary, where 
#   - `dTrHash[ word ] = index in the dTr`
def cutdict(dictionary, dictcounts, mincount=100):
    for cutoff in range(len(counts)):
        if counts[cutoff] < mincount:
            break
    print "Cutoff happens at index "+str(cutoff)
    dcut = dictionary[:cutoff]
    ccut = dictcounts[:cutoff]

    # Create a has for searching in the dictionary
    dHash={}
    for i in range(cutoff):
        dHash[ dcut[i] ] = i
    
    return dcut, ccut, dHash


# ## Make a list of the valid keys
# - `validkeys` is the list of image keys that are valid  
# - `validkeys_dict[ key ]: list of words`
def imid_to_array(image_ids):
    
    ii2ai={}
    for ai,ii in enumerate(image_ids):
        ii2ai[ii]=ai
        
    return ii2ai

# Obtain the valid commericials
def getvalid(image_ids, text_feats, dHash, id2array=None):
    '''
    Get the valid images. You will need:
    
    - image_ids:  image ids that you'd like to check
    - text_feats: the words that you'd checking against
    - id2array:   the conversion between image id and image array
                  this is necessary only if you want to do some conversion
    - dHash:      dictionary of words used
    
    Returns
    - valididx:   the valid indices in the image array 
    '''    
    
    validkeys = []
    valididxs = []
    validkeys_dict = {}
    for i,key in enumerate(image_ids):
        words = text_feats[key]
        new_words = []
        for word in set(words):
            if dHash.has_key(word):
                new_words+=[word]
        if len(new_words):
            validkeys += [key]
            if id2array:
                valididxs += [id2array[key]]
            validkeys_dict[key]=new_words

    print 'Using {} of the dataset'.format( len(validkeys)*1.0/alldata.image_feats.shape[0] )
    
    return valididxs, validkeys, validkeys_dict


# ## Make a one hot encoding given list of validkeys, its words, and a dictionary
def valid2onehot(vk, vk_dict, DHash):
    print "Creating one hot vectors"
    onehots = np.zeros((len(vk),len(DHash)))
    for i,j in enumerate(vk):
        words = vk_dict[j]
        for word in words:
            onehots[i, DHash[word] ] += 1
    return onehots


def getimlist( ids ):
    
    imginfofile = json.load(open('/data/fs4/datasets/vg_unzipped/image_data.json'))
    imginfo = {}
    for img in imginfofile:
        imginfo[img['id']] = img['url'].split('/')[-1]
    
    imlist = []
    for the_id in ids:
        imlist+= [imginfo[the_id]]
    
    return imlist


def getsplits( ids, arrayidxs, validkeys ):
    idxlist = []    
    keylist = []
    # Index into image array
    for i,(ai,ii) in enumerate(zip(arrayidxs, validkeys)):
        if ii in ids:
            idxlist+=[ai] 
            keylist+=[i]
    return idxlist,keylist


# Image indexes to arrays
id2array = imid_to_array(alldata.image_ids) 

# Get the vocab from the text features
vocab = getvocab( alldata.text_feats )

# Order the vocabulary in order of frequencies
D, counts = ordervocab(vocab)
dTr, _, dTrHash = cutdict(D,counts,mincount=100)
arrayidxs, validkeys, validkeys_dict = getvalid( alldata.image_ids, alldata.text_feats, dTrHash, id2array=id2array )
# onehots = valid2onehot( validkeys, validkeys_dict, dTrHash )

# Get the training, valid, and test lists
trainids = set( [str(imid) for imid in splits['train']] )
validids = set( [str(imid) for imid in splits['val']] )
testids  = set( [str(imid) for imid in splits['test']] )

def makeattalos(fname, ids, vkd, tvtsplit='train'):

    newfile = open(fname,'w')
    notfile = open(fname+'notfound','w')
    for im in ids:
        if vkd.has_key(str(im)):
            newfile.write('{}\t{}\t'.format( im+'.jpg',tvtsplit ) )
            comma=''
            for tag in vkd[str(im)]:
                newfile.write('{}{}'.format(comma,tag))
                comma=','
            newfile.write('\n')
        else:
            notfile.write("{} IS NOT FOUND\n".format(str(im)))
            
    newfile.close()
    notfile.close()

makeattalos('visgenome_training.txt', trainids, validkeys_dict)
makeattalos('visgenome_validation.txt', validids, validkeys_dict, tvtsplit='val')
makeattalos('visgenome_testing.txt', testids, validkeys_dict, tvtsplit='test')
