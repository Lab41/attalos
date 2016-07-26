#!/bin/bash

./word2vec -train README.txt -output outfile -binary 1 -window 5 -negative 5 -min-count 0 -iter 100 -size 50
./word2vec -train README.txt -cbow 0 -output outfile2 -binary 1 -window 5 -negative 5 -min-count 0 -iter 100 -read-vecs outfile -read-fix-vocab imdict2.txt -update-vout-vectors imvecs.txt -size 50
