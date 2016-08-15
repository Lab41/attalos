# Utility functions

negsamp.py - Negative sampling class

## progressbar
### Progress bar class

## readw2v
### Reading word2vec vectors

Using code https://code.google.com/archive/p/word2vec/, it is possible to create word vectors. These word vectors are typically optimally stored in binary since vectors are dense floats. this code will read those in. For example, if we are using the first 8 billion characters from Wikipedia (downloaded from the original code), after using the word2vec code, let's say it produces the file text8b.bin. Then the following will read that into a dictionary:

```
import attalos.imgtxt_algorithms.util.readw2v as rw2v
F = rw2v.ReadW2V('text8B.bin')
vectors = F.readlines()
```
