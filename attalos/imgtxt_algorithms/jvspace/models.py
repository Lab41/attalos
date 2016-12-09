import numpy as np
import tensorflow as tf
import tflearn


## NUMPY rewrites
############################################################
def sigmoid( x ):
    if x.shape:
        x[x>500.0]=500.0
        x[x<-500.0]=-500.0
    else:
        if x>500.0:
            x=500.0
        if x<-500.0:
            x=-500.0
    return 1.0 / (1 + np.exp(-x))


## IMAGE COST FUNCTIONS
############################################################
# Negative samples and loss (3D tensor function)
def meanlogsig_3d( f, V ):
    return tf.reduce_mean( tf.log( tf.sigmoid( tf.reduce_sum(f*V, reduction_indices=2) ) ) )
def w2vloss( f, pVecs, nVecs ):
    tfpos = meanlogsig_3d(f, pVecs)
    tfneg = meanlogsig_3d(-f, nVecs)
    return -(tfpos + tfneg)

# Sum of word vectors (2D tensor function)
def meanlogsig_2d( f, V ):
    return tf.reduce_mean( tf.log( tf.sigmoid( tf.reduce_sum( f*V, reduction_indices=1 ) ) ) )
def w2vsumloss( f, pVec, nVec ):
    return -( meanlogsig_2d(f, pVec) + meanlogsig_2d(-f, nVec) )


## NEURAL NETWORK MODELS
############################################################
# Cross-correlation image model
def imageXmodel(input_size=2048, vec_size=200, hidden_units=[]):
    '''
    imagemodel( input_size, vec_size, hidden_units )
    '''
    pVecs = tf.placeholder(tf.float32, shape=[None, None, vec_size], name='pvecs')
    nVecs = tf.placeholder(tf.float32, shape=[None, None, vec_size], name='nvecs')    
    inputs = tf.placeholder(tf.float32, shape=[None, input_size], name='input')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # Iterate through the hidden units list and connect the graph
    layer_i = inputs
    for i, hidden in enumerate(hidden_units):
        # layer_i = tflearn.fully_connected(layer_i, hidden, activation='sigmoid', name='fc'+str(i))
        layer_i = tf.contrib.layers.relu(layer_i, hidden)
        # layer_i = tflearn.layers.normalization.batch_normalization(layer_i)
        layer_i = tf.contrib.layers.batch_norm(layer_i)
    # prediction = tflearn.fully_connected(layer_i, vec_size, activation='sigmoid', name='output')
    # prediction = tflearn.fully_connected(layer_i, vec_size, activation='linear', name='output')
    prediction = tf.contrib.layers.linear(layer_i, vec_size)
    
    # Loss function and optimizer to be used
    loss = w2vloss(prediction,pVecs,nVecs)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            
    # Return actual variables
    return inputs, pVecs, nVecs, prediction, loss, optimizer, learning_rate

# Sum of word vectors image model
def imagesummodel(input_size=2048, vec_size=200, hidden_units=[]):
    '''
    imagesummodel( input_size, vec_size, hidden_units )
    '''
    pVecs = tf.placeholder(tf.float32, shape=[None, None, vec_size], name='pvecs')
    nVecs = tf.placeholder(tf.float32, shape=[None, None, vec_size], name='nvecs')    
    inputs = tf.placeholder(tf.float32, shape=[None, input_size], name='input')

    meanP = tf.reduce_mean(pVecs, reduction_indices=0)
    meanN = tf.reduce_mean(nVecs, reduction_indices=0)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # Iterate through the hidden units list and connect the graph
    layer_i = inputs
    for i, hidden in enumerate(hidden_units):
        layer_i = tflearn.fully_connected(layer_i, hidden, activation='relu', name='fc'+str(i))
    prediction = tflearn.fully_connected(layer_i, vec_size, activation='linear', name='output')

    # Loss function and optimizer to be used
    loss = w2vsumloss(prediction,meanP,meanN)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            
    # Return actual variables
    return inputs, pVecs, nVecs, prediction, loss, optimizer, learning_rate

# Negative samples and loss (3D tensor function)
#
# Let B be the batch size, N be the number of positive/negative samples, and d be the feature dimension
#
# Input C: 
#    batch_size x neg_samps x vocabulary_sample_size
# Feature input from neural network f: 
#    batch_size x ftr_dimensions
# Input V_o:
#    neg_samps x ftr_dimensions
def wordloss( C, f, V ):
    if type(C) == np.ndarray:
        dots = C*np.log(sigmoid(f.dot(V.T)))+(1-C)*np.log(1-sigmoid(f.dot(V.T)))
        return -dots.mean()
    else:
        dots = C*tf.log(tf.sigmoid(tf.matmul(f,V)))+(1-C)*tf.log(tf.sigmoid(-tf.matmul(f,V)))
        return -tf.reduce_mean(dots)

def w2vloss( f, pVecs, nVecs ):
    tfpos = meanlogsig_3d(f, pVecs)
    tfneg = meanlogsig_3d(-f, nVecs)
    return -(tfpos + tfneg)

def imageWmodel(input_size=2048, vec_size=200, hidden_units=[]):
    '''
    imagemodel( input_size, vec_size, hidden_units )
    '''
    pVecs = tf.placeholder(tf.float32, shape=[None, None, vec_size], name='pvecs')
    nVecs = tf.placeholder(tf.float32, shape=[None, None, vec_size], name='nvecs')    
    wVecs = tf.placeholder(tf.float32, shape=[None, vec_size], name='wvecs')
    CorrW = tf.placeholder(tf.float32, shape=[None, None, None], name='word_correlations')
    inputs = tf.placeholder(tf.float32, shape=[None, input_size], name='input')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    # Iterate through the hidden units list and connect the graph
    layer_i = inputs
    for i, hidden in enumerate(hidden_units):
        layer_i = tflearn.fully_connected(layer_i, hidden, activation='relu', name='fc'+str(i))
    prediction = tflearn.fully_connected(layer_i, vec_size, activation='sigmoid', name='output')

    # Loss function and optimizer to be used
    imloss = w2vloss(prediction,pVecs,nVecs)
    wdloss = wordloss( CorrW, prediction, tf.transpose(wVecs) )
    loss = imloss+wdloss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
            
    # Return actual variables
    return inputs, pVecs, nVecs, wVecs, CorrW, prediction, imloss, wdloss, loss, optimizer, learning_rate


## UPDATE LAST LAYER (VO)
############################################################
# If using sum of word vectors, use updateVoSum
def updateVoSum(vin, pVecs, nVecs, vpindex, vnindex, Vo, learnrate=0.01):
    pVecs = pVecs.mean(axis=0)
    nVecs = nVecs.mean(axis=0)
    vinp = ( (1 - sigmoid(( pVecs*vin).sum(axis=1)))*vin.T ).T
    vinn = - ( sigmoid(( nVecs*vin).sum(axis=1))*vin.T ).T
    for i, (vpi, vni) in enumerate(zip(vpindex,vnindex)):
        Vo[ vpindex[i] ] += learnrate*vinp[i]
        Vo[ vnindex[i] ] += learnrate*vinn[i]
    return Vo
# If using X-correlation, use updateVoX
def updateVoX(vin, pVecs, nVecs, vpindex, vnindex, Vo, learnrate=0.01):
    for i, (vpi, vni) in enumerate(zip(vpindex, vnindex)):
        Vo[vpi]+=learnrate*np.outer(1 - sigmoid(Vo[vpi].dot(vin[i])),vin[i])
        Vo[vni]-=learnrate*np.outer(sigmoid(Vo[vni].dot(vin[i])),vin[i])
    return Vo

# def updateWordX(vin, pVecs, nVecs, vpindex, vnindex, VoU, Cmat, learnrate=0.01):
#     for i, (vpi, vni) in enumerate(zip(vpindex, vnindex)):
#         VoU[vpi]+=0.01*np.outer(1 - sigmoid(Vo[vpi].dot(vin[i])),vin[i])
#         VoU[vni]-=0.01*np.outer(sigmoid(Vo[vni].dot(vin[i])),vin[i])
#     return VoU


## GET IMAGE BATCH
############################################################
# Get batch from randomly sampling from one hots
def get_batch(pBatch, Vo, numSamps=[5,10]):
    
    nBatch = 1.0 - pBatch
    # pVecs = pBatch.dot(Vo)
    # nVecs = nBatch.dot(Vo)
    
    Vpbatch = np.zeros((len(pBatch), numSamps[0], 200))
    Vnbatch = np.zeros((len(nBatch), numSamps[1], 200))
    vpia = []; vnia = [];
    for i,unisamp in enumerate(pBatch):
        vpi = np.random.choice( range(len(unisamp)) , size=numSamps[0],  p=1.0*unisamp/sum(unisamp))
        Vpbatch[i] = Vo[ vpi ]
        vpia += [vpi]
        
    for i,unisamp in enumerate(nBatch):
        vni = np.random.choice( range(len(unisamp)) , size=numSamps[1], p=1.0*unisamp/sum(unisamp))
        Vnbatch[i] = Vo[ vni ]
        vnia += [vni]
    
    Vpbatch = Vpbatch.transpose(1,0,2)
    Vnbatch = Vnbatch.transpose(1,0,2)
    
    return Vpbatch, Vnbatch, vpia, vnia
# Get batch from randomly sample from one hots but single images
def get_batch_image(pBatch, nBatch, Vo, numSamps=[5,10]):
    
    # nBatch = 1.0 - pBatch
    # pVecs = pBatch.dot(Vo)
    # nVecs = nBatch.dot(Vo)
    
    Vpbatch = np.zeros((len(pBatch), numSamps[0], 200))
    Vnbatch = np.zeros((len(nBatch), numSamps[1], 200))
    vpia = []; vnia = [];
    for i,unisamp in enumerate(pBatch):
        vpi = np.random.choice( range(len(unisamp)) , size=numSamps[0],  p=1.0*unisamp/sum(unisamp))
        Vpbatch[i] = Vo[ vpi ]
        vpia += [vpi]
        
    for i,unisamp in enumerate(nBatch):
        vni = np.random.choice( range(len(unisamp)) , size=numSamps[1], p=1.0*unisamp/sum(unisamp))
        Vnbatch[i] = Vo[ vni ]
        vnia += [vni]
    
    Vpbatch = Vpbatch.transpose(1,0,2)
    Vnbatch = Vnbatch.transpose(1,0,2)
    
    return Vpbatch, Vnbatch, vpia, vnia


## WORD OPTIMIZATION
############################################################
# word parameter optimization
wordlr = 1e-4

# Nonlinear optimization
def costYViVo(Y, Vi, Vo):
    xcorrs = Vi.dot(Vo.T)
    fullcost = Y*np.log(sigmoid( xcorrs )) + (1-Y)*np.log(sigmoid(1-xcorrs))
    return -fullcost.mean()

def costYXcorr(Y, xcorrs):
    fullcost = Y*np.log(sigmoid( xcorrs )) + (1-Y)*np.log(sigmoid(1-xcorrs))
    return -fullcost.mean()

# New full vectors. Assumes that Vo is optimized through the image space
def adaptWords( VoD, VoUnew, Cmat, wordlr=1.0e-4 ):
    xCorrs = VoD.dot(VoUnew.T)
    # b4cost= costYXcorr(Cmat, xCorrs)
    VoD+= (wordlr*Cmat*(1-sigmoid(xCorrs))).dot(VoUnew)
    VoD+= (wordlr*(Cmat-1)*(1-sigmoid(1-xCorrs))).dot(VoUnew)
    return VoD



