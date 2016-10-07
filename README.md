# Attalos

This project explores methods in constructing a common representation across modalities using unstructured and heterogeneous data collections. This representation will be a joint vector space that can be used to compare concepts in a numerical manner, no matter how different the modality or type of concepts are. The type of data can vary widely and will range from images to text to social structure, but comparisons between them will be seamless and can be made with Euclidean operations. In other words, concepts that are proximal in the joint vector space will be exhibit semantic similarity. Lab41 will evaluate the vector space using well-known metrics on classification tasks.

# Getting Started

To start downloading datasets, see [the Dataset README](attalos/dataset/README.md).

To learn how to preprocess data, see [the Preprocessing READE](attalos/preprocessing/README.md).

To learn about running the performance metrics, see [the Evaluation README](attalos/dataset/README.md).

To learn how to optimize wordvectors, see [the Update-words README](attalos/imgtxt_algorithms/updatewords/README.md).

To learn how to run the demo app, see [the Demo README](attalos/imgtxt_algorithms/demo_app/README.md).

To learn about our utilities classes, see [the Util README](attalos/imgtxt_algorithms/util/README.md).


# Install Instructions

```
git clone https://github.com/Lab41/attalos.git
cd attalos
make
```

# Required Dependencies

- Docker
- make

# Contributing to Attalos

Want to contribute?  Awesome!  Issue a pull request or see more details [here](https://github.com/Lab41/attalos/blob/master/CONTRIBUTING.md).
