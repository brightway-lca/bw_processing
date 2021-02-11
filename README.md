# bw_processing

Library for storing numeric data for use in matrix-based calculations. Designed for use with the [Brightway life cycle assessment framework](https://brightway.dev/).

[![Azure Status](https://dev.azure.com/mutel/Brightway%20CI/_apis/build/status/brightway-lca.bw_processing?branchName=master)](https://dev.azure.com/mutel/Brightway%20CI/_build/latest?definitionId=7&branchName=master) [![Travis Status](https://travis-ci.org/brightway-lca/bw_processing.svg?branch=master)](https://travis-ci.org/brightway-lca/bw_processing) [![Coverage Status](https://coveralls.io/repos/github/brightway-lca/bw_processing/badge.svg?branch=master)](https://coveralls.io/github/brightway-lca/bw_processing?branch=master) [![Documentation](https://readthedocs.org/projects/bw-processing/badge/?version=latest)](https://bw-processing.readthedocs.io/en/latest/)

## Table of Contents

- [Background](#background)
- [Concepts](#concepts)
- [Install](#install)
- [Usage](#usage)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [License](#license)

## Background

The [Brightway LCA framework](https://brightway.dev/) has stored data used in constructing matrices in binary form as numpy arrays for years. This package is an evolution of that approach, and adds the following features:

* **Consistent names for row and column fields**. Previously, these changed for each matrix, to reflect the role each row or column played in the model. Now they are always the same for all arrays (`"row"` and `"col"`), making the code simpler and easier to use.
* **Provision of metadata**. Numpy binary files are only data - `bw_processing` also produces a metadata file following the [data package standard](https://specs.frictionlessdata.io/data-package/). Things like data license, version, and unique id are now explicit and always included.
* **Support for vector and array data**. Vector (i.e. only one possible value per input) and array (i.e. many possible values, also called presamples) data are now both natively supported in data packages.
* **Portability**. Processed arrays can include metadata that allows for reindexing on other machines, so that processed arrays can be distributed and reused. Before, this was not possible, as integer IDs were randomly assigned on each computer, and would be different from machine to machine or even across Brightway projects.
* **Dynamic data sources**. Instead of requiring that data for matrix construction be present and savedd on disk, it can now be generated dynamically, either through code running locally or on another computer system. This is a big step towards embeddding life cycle assessment in a web of environmental models.
* **Use [PyFilesystem2](https://docs.pyfilesystem.org/en/latest/) for file IO**. The use of this library allows for data packages to be stored on your local computer, or on [many logical or virtual file systems](https://docs.pyfilesystem.org/en/latest/guide.html).
* **Simpler handling of numeric values whose sign should be flipped**. Sometimes it is more convenient to specify positive numbers in dataset definitions, even though such numbers should be negative when inserted into the resulting matrices. For example, in the technosphere matrix in life cycle assessment, products produced are positive and products consumed are negative, though both values are given as positive in datasets. Brightway used to use a type mapping dictionary to indicate which values in a matrix should have their sign flipped after insertion. Such mapping dictionaries are brittle and inelegant. `bw_processing` uses an optional boolean vector, called `flip`, to indicate if any values should be flipped.
* **Separation of uncertainty distribution parameters from other data**. Fitting data to a [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (PDF), or an estimate of such a PDF, is only one approach to quantitative uncertainty analysis. We would like to support other approaches, including [direct sampling from real data](https://github.com/PascalLesage/presamples/). Therefore, uncertainty distribution parameters are stored separately,  only loaded if needed, and are only one way to express quantitative uncertainty.

## Concepts

### Data packages

Data objects can be vectors or arrays. Vectors will always produce the same matrix, while arrays have multiple possible values for each element of the matrix. Arrays are a generalization of the [presamples library](https://github.com/PascalLesage/presamples/).

### Data needed for matrix construction

### Vectors versus arrays

### Persistent versus dynamic

Persistent data is fixed, and can be completely loaded into memory and used directly or written to disk. Dynamic data is only resolved as the data is used, during matrix construction and iteration. Dynamic data is provided by *interfaces* - Python code that either generates the data, or wraps data coming from other software. There are many possible use cases for data interfaces, including:

* Data that is provided by an external source, such as a web service
* Data that comes from an infinite python generator
* Data from another programming language
* Data that needs processing steps before it can be directly inserted into a matrix

Only the actual numerical values entered into the matrix is dynamic - the matrix index values (and optional flip vector) are still static, and need to be provided as Numpy arrays when adding dynamic resources.

Interfaces must implement a simple API. Dynamic vectors must support the python generator API, i.e. implement `__next__()`.

Dynamic arrays must pretend to be Numpy arrays, in that they need to implement `.shape` and `.__getitem__(args)`.

* `.shape` must return a tuple of two integers. The first should be the number of elements returned, though this is not used. The second should be the number of columns available - an integer. This second value can also be `None`, if the interface is infinite.
* `.__getitem__(args)` must return a one-dimensional Numpy array corresponding to the column `args[1]`. This method is called when one uses code like `some_array[: 20]`. In our case, we will always take all rows (the `:`), so the first value can be ignored.

Here are some example interfaces (also given in `bw_processing/examples/interfaces.py`):

```python
import numpy as np


class ExampleVectorInterface:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.size = self.rng.integers(2, 10)

    def __next__(self):
        return self.rng.random(self.size)


class ExampleArrayInterface:
    def __init__(self):
        rng = np.random.default_rng()
        self.data = rng.random((rng.integers(2, 10), rng.integers(2, 10)))

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, args):
        if args[1] >= self.shape[1]:
            raise IndexError
        return self.data[:, args[1]]
```

### Policies

Data package policies define how the data should be used. Policies apply to the entire data package; you may wish to adjust what is stored in which data packages to get the effect you desire.

There are two policies that apply to all data resources:

**sum_intra_duplicates** (default `True`): What to do if more than one data point for a given matrix element is given in each *vector or array resource*. If true, sum these values; otherwise, the last value provided is used.

**sum_inter_duplicates** (default: `False`): What to do if data from a given resource overlaps data already present in the matrix. If true, add the given value to the existing value; otherwise, the existing values will be overwritten.

There are three policies that apply only to array data resources, where a different column from the array is used in matrix construction each time the array is iterated over:

**combinatorial** (default `False`): If more than one array resource is available, this policy controls whether all possible combinations of columns are guaranteed to occur. If `combinatorial` is `True`, we use [`itertools.combinations`](https://docs.python.org/3/library/itertools.html#itertools.combinations) to generate column indices for the respective arrays; if `False`, column indices are either completely random (with replacement) or sequential.

Note that you will get `StopIteration` if you exhaust all combinations when `combinatorial` is `True`.

Note that `combinatorial` cannot be `True` if infinite array interfaces are present.

**sequential** (default `False`): Array resources have multiple columns, each of which represents a valid system state. Default behaviour is to choose from these columns at random (including replacement), using a RNG and the data package `seed` value. If `sequential` is `True`, columns in each array will be chosen in order starting from column zero, and will rewind to zero if the end of the array is reached.

Note that if `combinatorial` is `True`, `sequential` is ignored; instead, the column indices are generated by [`itertools.combinations`](https://docs.python.org/3/library/itertools.html#itertools.combinations).

Please make sure you understand how `combinatorial` and `sequential` interact! There are three possibilities:

* `combinatorial` and `sequential` are both `False`. Columns are returned at random.

* `combinatorial` is `False`, `sequential` is `True`. Columns are returned in increasing numerical order without any interaction between the arrays.

* `combinatorial` is `True`, `sequential` is ignored: Columns are returned in increasing order, such that all combinations of the different array resources are provided. `StopIteration` is raised if you try to consume additional column indices.

## Install

Install using pip or conda (channel `cmutel`). Depends on `numpy` and `pandas` (for reading and writing CSVs).

Has no explicit or implicit dependence on any other part of Brightway.

## Usage

The main interface for using this library is the `Datapackage` class. However, instead of creating an instance of this class directly, you should use the utility functions `create_datapackage` and `load_datapackage`.

A datapackage is a set of file objects (either in-memory or on disk) that includes a metadata file object, and one or more data resource files objects. The metadata file object includes both generic metadata (i.e. when it was created, the data license) and metadata specific to each data resource (how it can be used in calculations, its relationship to other data resources). Datapackages follow the [data package standard](https://specs.frictionlessdata.io/data-package/).

### Creating datapackages

Datapackages are created using `create_datapackage`, which takes the following arguments:

* dirpath: `str` or `pathlib.Path` object. Where the datapackage should be saved. `None` for in-memory datapackages.
* name: `str`: The name of the overall datapackage. Make it meaningful to you.
* id_: `str`, optional. A unique id for this package. Automatically generated if not given.
* metadata: `dict`, optional. Any additional metadata, such as license and author.
* overwrite: `bool`, default `False`. Overwrite an existing resource with the same `dirpath` and `name`.
* compress: `bool`, default `False`. Save to a zipfile, if saving to disk.

Calling this function return an instance of `Datapackage`. You still need to add data.

## Contributing

Your contribution is welcome! Please follow the [pull request workflow](https://guides.github.com/introduction/flow/), even for minor changes.

When contributing to this repository with a major change, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository.

Please note we have a [code of conduct](https://github.com/brightway-lca/bw_processing/blob/master/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

### Documentation and coding standards

* [Black formatting](https://black.readthedocs.io/en/stable/)
* [Semantic versioning](http://semver.org/)

## Maintainers

* [Chris Mutel](https://github.com/cmutel/)

## License

[BSD-3-Clause](https://github.com/brightway-lca/bw_processing/blob/master/LICENSE). Copyright 2020 Chris Mutel.
