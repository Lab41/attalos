# Attalos

This project explores methods in constructing a common representation across modalities using unstructured and heterogeneous data collections. This representation will be a joint vector space that can be used to compare concepts in a numerical manner, no matter how different the modality or type of concepts are. The type of data can vary widely and will range from images to text to social structure, but comparisons between them will be seamless and can be made with Euclidean operations. In other words, concepts that are proximal in the joint vector space will be exhibit semantic similarity. Lab41 will evaluate the vector space using well-known metrics on classification tasks.


# Getting Started

Nothing to see here yet, move along...

# Install Instructions

```
git clone https://github.com/Lab41/attalos.git
cd attalos
make
```

# Required Dependencies

- Docker
- make

# Usage Examples

```
make run
```

# Documentation
- [Docs](https://github.com/Lab41/attalos/tree/master/docs)

# Tests

Tests are currently written in py.test for Python.  The tests are automatically run when building the dockerfile.

They can also be tested using:
```
make test
```

# Contributing to Attalos

Want to contribute?  Awesome!  Issue a pull request or see more details [here](https://github.com/Lab41/attalos/blob/master/CONTRIBUTING.md).
