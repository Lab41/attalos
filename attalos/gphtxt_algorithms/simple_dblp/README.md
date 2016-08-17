# Simple Graph and Text Embedding Using Word2Vec

Run demo_dblp.sh run code on dblp. This code will

1. Download the dblp.xml file from "http://dblp.uni-trier.de/xml/". Check the date to see how recent the dump was.
2. Parse the XML file in Python
3. Postprocess by removing periods and words
4. Compile the files in attalos/attalos/imgtxt_algorithms/updatewords
5. Run word2phrase and word2vec on authors and words
