#!/bin/bash

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
  echo "USAGE: rundblp.sh <PATH-TO-ATTALOS> <NUM-PAPERS, default=3000000>"
  exit
fi
if [ $# -eq 1 ]; then
  numrecords=3000000
fi

PATH_TO_ATTALOS=$1
# Download if necessary
if [ ! -e dblp.xml ]; then
    wget http://dblp.uni-trier.de/xml/dblp.xml.gz
    gunzip dblp.xml.gz 
fi

# Run parser to get the first 300000 papers from dblp
echo "Running parser of XML into temp.txt"
python dblparse.py $numrecords > temp.txt

# Make everything lowercase and remove periods
echo "Making everything lowercase"
awk '{print tolower($0)}' temp.txt | sed 's/\.//g' > temp2.txt

# Remove everything that's only a character long
echo "Removing names that are only one letter long"
python remove1char.py temp2.txt > authortitles.txt

echo "Compiling word2vec code (Mikolov)"
export PATH=$PATH:$PATH_TO_ATTALOS/attalos/imgtxt_algorithms/updatewords
CURDIR=$PWD
cd $PATH_TO_ATTALOS/attalos/imgtxt_algorithms/updatewords/
make
cd $CURDIR

# Run "word2vec" and "word2phrase"
echo "Running word2vec"
word2phrase -train authortitles.txt -output authortitles_phrase.txt -min-count 5
word2vec -train authortitles_phrase.txt -output authortitles_phrase.bin -size 50 -window 20 -sample 1e-4 -negative 15 -hs 0 -binary 1 -cbow 1 -iter 30
mv authortitles_phrase.bin.vin.bin authortitles.bin
distance authortitles.bin

# Cleanup
rm temp*.txt
rm *.vout.*
rm authortitles.txt
