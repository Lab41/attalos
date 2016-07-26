import struct
import array
import numpy as np

# Read word2vec from Google's word2vec C code into Python dictionary
class ReadW2V():

    MAXSTRING = 1000

    def __init__(self, filename):
        
        self.fd = open(filename, 'rb')
        self.numvecs = int(self.readstring())
        self.numdims = int(self.readstring())
        self.vectors = {}

    def readstring(self):

        word=''

        for i in xrange(self.MAXSTRING):
            char = self.fd.read(1)
            if char.isspace():
                break
            else:
                word+=char

        return word

    def readfloats(self, num2read):

        arr = array.array('f')
        arr.read(self.fd, num2read)
        return np.array( arr )

    def readfloat(self):

        return struct.unpack('f', self.fd.read(4) )

    def readline(self):
        
        word = self.readstring()
        vec = self.readfloats(self.numdims)
        self.readstring()

        return (word,vec)

    def readlines(self, num2read=0):

        if num2read==0:
            num2read=self.numvecs

        for i in xrange(num2read):
            word, vec =  self.readline()
            self.vectors[word] = vec

        return self.vectors

    def vec2mat(self):

        return np.array(self.vectors.values())

    def words(self):

        return self.vectors.keys()


