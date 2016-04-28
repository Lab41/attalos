#!/bin/bash

source dblpenv/bin/activate
export PATH=$PATH:$PWD/w2v

word2phrase -train at_nochar.txt -output ai_nochar_phrase.txt -min-count 5
word2vec -train ai_nochar_phrase.txt -output at_nochar_phrase.bin -size 50 -window 20 -sample 1e-4 -negative 15 -hs 0 -binary 1 -cbow 1 -iter 30
