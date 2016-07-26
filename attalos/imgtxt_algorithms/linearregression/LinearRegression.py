
# import theano.tensor as T
# import theano
# import tensorflow as tf
import numpy as np
from collections import OrderedDict
import matplotlib.pylab as plt

class LinearRegression():
    
    def __init__(self, dims=(4096,291), normX = False, normY = False):

        self.normX = normX
        self.normY = normY
                
        self.W = np.zeros( dims )
            
    def train(self, X, Y):
        
        if self.normX:
            X = self.normfeats( X ).astype(np.float32)
        if self.normY:
            Y = self.normfeats( Y ).astype(np.float32)
            
        self.W = self.inverseproblem( X , Y )
    
    # ## Functional definition
    def normfeats(self, X ):
        return ( X.T / np.linalg.norm( X, axis = 1 ) ).T

    def buildcov( self, X ):
        if type(X)==np.ndarray:
            return X.T.dot(X)
        return tf.matmul(tf.transpose(X),X)

    # Pseudoinverse
    def invertcov( self, X ):
        return np.linalg.inv( self.buildcov(X) )


    # Pseudoinverse: return W = Y \ X
    def inverseproblem( self, X, Y ):
        print "Building W matrix = Y \ X = Y^T X (X X^T)^-1"
        return Y.T.dot(X).dot( self.invertcov(X) )

    # Given a matrix from "buildMat", build a predictor
    def predict( self, X ):
        Xn = self.normfeats( X )
        return self.W.dot( Xn.T.astype(np.float32) ).T

