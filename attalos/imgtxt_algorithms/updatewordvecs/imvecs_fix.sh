#!/bin/bash

./word2vec -train text8 -output iaprtc12-text8 -binary 1 -window 10 -negative 20 -min-count 0 -iter 100 -size 200 -update-vout-vectors imvecs.txt -threads 24
