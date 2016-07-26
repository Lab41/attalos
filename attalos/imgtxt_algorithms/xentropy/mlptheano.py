import theano.tensor as T
import theano
import numpy as np
from collections import OrderedDict

# Define the function
x = T.matrix()
y = T.matrix()
W = theano.shared(np.random.ranf((5,5)))
b = theano.shared(np.random.ranf(5))
yh = (W.dot(x).T + b).T

# Define the gradient updates
loss = T.sqr(y - yh).mean()
params = [W, b]
gW, gb = T.grad(loss, params)
sgd = OrderedDict( { W: W - gW, b: b-gb } )

# Compile to theano functionality
optfxn = theano.function( [x,y], outputs=[yh, loss], updates=sgd, allow_input_downcast=True )

fxn = theano.function( inputs =[x], outputs = yh )
yy = fxn( np.random.ranf((5,100)) )

