# bw_processing

Library for storing numeric data for use in matrix-based calculations. Designed for use with the [Brightway life cycle assessment framework](https://brightway.dev/).

[![Azure Status](https://dev.azure.com/mutel/Brightway%20CI/_apis/build/status/brightway-lca.bw_processing?branchName=master)](https://dev.azure.com/mutel/Brightway%20CI/_build/latest?definitionId=7&branchName=master) [![Travis Status](https://travis-ci.org/brightway-lca/bw_processing.svg?branch=master)](https://travis-ci.org/brightway-lca/bw_processing) [![Coverage Status](https://coveralls.io/repos/github/brightway-lca/bw_processing/badge.svg?branch=master)](https://coveralls.io/github/brightway-lca/bw_processing?branch=master) [![Documentation](https://readthedocs.org/projects/bw-processing/badge/?version=latest)](https://bw-processing.readthedocs.io/en/latest/)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [License](#license)

## Background

The [Brightway LCA framework](https://brightway.dev/) has stored data used in constructing matrices in binary form as numpy arrays for years. This package is an evolution of that approach, and adds the following features:

* **Consistent names for row and column fields**. Previously, these changed for each matrix, to reflect the role each row or column played in the model. Now they are always the same for all arrays (`"row"` and `"col"`), making the code simpler and easier to use.
* **Provision of metadata**. Numpy binary files are only data - `bw_processing` also produces a metadata file following the [data package standard](https://specs.frictionlessdata.io/data-package/). Things like data license, version, and unique id are now explicit and always included.
* **Support for vector and array data**. Vector (i.e. only one possible value per input) and array (i.e. many possible values, also called presamples) data are now both natively supported in data packages.
* **Portability**. Processed arrays can include metadata that allows for reindexing on other machines, so that processed arrays can be distributed and reused. Before, this was not possible, as integer IDs were randomly assigned on each computer, and would be different from machine to machine or even across Brightway projects.
* **Dynamic data sources**. Data for matrix construction (both vectors and arrays) can be provided on demand, using a defined API. This allows for data sources on the cloud, or other external data interfaces.
* **Use [PyFilesystem2](https://docs.pyfilesystem.org/en/latest/) for file IO**. The use of this library allows for data packages to be specified locally on disk, but also in memory, or on many cloud services or network resources.
* **Simpler handling of numeric values whose sign should be flipped**. Sometimes it is more convenient to specify positive numbers in dataset definitions, even though such numbers should be negative when inserted into the resulting matrices. For example, in the technosphere matrix in life cycle assessment, products produced are positive and products consumed are negative, though both values are given as positive in datasets. Brightway used to use a type mapping dictionary to indicate which values in a matrix should have their sign flipped after insertion. Such mapping dictionaries are brittle and inelegant. `bw_processing` uses an optional boolean vector, called `flip`, to indicate if any values should be flipped.
* **Separation of uncertainty distribution parameters from other data**. Fitting data to a [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (PDF), or an estimate of such a PDF, is only one approach to quantitative uncertainty analysis. We would like to support other approaches, including [direct sampling from real data](https://github.com/PascalLesage/presamples/). Therefore, uncertainty distribution parameters are stored separately, and only loaded if needed.

## Install

Install using pip or conda (channel `cmutel`). Depends on `numpy` and `pandas` (for reading and writing CSVs).

Has no explicit or implicit dependence on any other part of Brightway.

## Usage

`bw_processing` uses a hierarchical structure: Data is stored in data packages, and inside each data package is a series of groups, each of which consists of data objects that define one part of a matrix.

Data objects can be vectors or arrays. Vectors will always produce the same matrix, while arrays have multiple possible values for each element of the matrix. Arrays are a generalization of the [presamples library](https://github.com/PascalLesage/presamples/).

Data objects can also be either persistent or dynamic. Persistent arrays can be saved, and are always the same, while dynamic arrays are resolved at run time, and can change every time they are accessed.

Probability distribution functions can only be defined for persistent vectors. Dynamic data can have uncertainty, but the dynamic data generator is responsible for modeling that uncertainty. Static arrays can't have uncertainty, as each column in the array represents the uncertainty and/or variability in the system performance or configuration.

Because this library supports multiple use cases, there are several dimensions to be aware of:

* Static versus dynamic data: Static data can be completely loaded into memory and used directly or written to disk. Dynamic data is only resolved as the data is used (i.e. during matrix construction, not during package creation), and can be infinite generators.
* Creating versus loading. In-memory arrays can only be created and then used, but not loaded. On-disk arrays, on the other hand, must be created, stored, and then loaded to be used. This is because saving a package to disk flushes the data from memory.
* Vectors versus arrays. Each element in a vector can only have one possible value - this is the same as a static matrix. Arrays have the same number of rows as the length of a vector, but multiple possible values of each parameter. Each column is therefore one conssitent possible set of parameter values. See the [presample library for more information](https://github.com/PascalLesage/presamples/).
* Probability distributions.

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

### Adding normal matrix data

Let's see it in action:

```python
In [1]: import bw_processing as bwp

In [2]: dp = bwp.Datapackage.load(bwp.examples_dir / "simple.zip")

In [3]: dp.metadata
Out[3]:
{'profile': 'data-package',
 'name': 'simple',
 'id': 'e64da3c213e44a589087fb1f41429fdf',
 'licenses': [{'name': 'ODC-PDDL-1.0',
   'path': 'http://opendatacommons.org/licenses/pddl/',
   'title': 'Open Data Commons Public Domain Dedication and License v1.0'}],
 'resources': [{'profile': 'data-resource',
   'format': 'npy',
   'mediatype': 'application/octet-stream',
   'name': 'simple-data-source',
   'matrix': 'technosphere',
   'path': 'simple-data-source.npy'}],
 'created': '2020-09-25T14:01:43.335137Z'}

In [4]: dp.data
Out[4]:
[array([( 1,  2, 2147483647, 2147483647, 0,  3.,  3., nan, nan, nan, nan, False, False),
        (14, 15, 2147483647, 2147483647, 0, 16., 16., nan, nan, nan, nan, False, False)],
       dtype=[('row_value', '<i4'), ('col_value', '<i4'), ('row_index', '<i4'), ('col_index', '<i4'), ('uncertainty_type', 'u1'), ('amount', '<f4'), ('loc', '<f4'), ('scale', '<f4'), ('shape', '<f4'), ('minimum', '<f4'), ('maximum', '<f4'), ('negative', '?'), ('flip', '?')])]
```

In this example, we returned the processed data package as a dictionary in memory, but normally `bw_processing` is used to persist data to disk.

There is also a utility function, `load_calculation_package`, which loads a saved calculation package in the same format as was returned by the example.

See the [documentation](https://bw-processing.readthedocs.io/en/latest/) for more information on how to use `bw_processing` to load, save, and use data.

### Policies

Data packages have specify four different policies that define how their data will be used. Each of these policies is global across the data package; you may wish to adjust what is stored in which data packages to get the effect you desire.

**combinatorial** (default `False`): If more than one array resource is available, this policy controls whether all possible combinations of columns are guaranteed to occur. If `combinatorial` is `True`, we use [`itertools.combinations`](https://docs.python.org/3/library/itertools.html#itertools.combinations) to generate column indices for the respective arrays; if `False`, column indices are either completely random (with replacement) or sequential.

Note that you will get `StopIteration` if you exhaust all combinations when `combinatorial` is `True`.

Note that `combinatorial` cannot be `True` if infinite interfaces are present.

**sequential** (default `False`): Array resources have multiple columns, each of which represents a valid system state. Default behaviour (`sequential` is `False`) is to choose from these columns at random (including replacement), using a RNG and the data package `seed` value. If `sequential` is `True`, columns in each array will be yielded in order starting from column zero.

Note that if `combinatorial` is `True`, `sequential` is ignored; instead, the column indices are generated by [`itertools.combinations`](https://docs.python.org/3/library/itertools.html#itertools.combinations).

Please make sure you understand how `combinatorial` and `sequential` interact! There are three possibilities:

* `combinatorial` and `sequential` are both `False`. Columns are returned completely randomly.

* `combinatorial` is `False`, `sequential` is `True`. Columns are returned in increasing numberical order without any interaction between the arrays.

* `combinatorial` is `True`, `sequential` is ignored: Columns are returned in increasing order, such that all combinations of the different array resources are provided. `StopIteration` is raised if you try to consume additional column indices.

**sum_duplicates** (default `False`): If more than one data point for a given matrix element is given in the data package (not necessarily in the same data resource), sum these values. If `sum_duplicates` is `False`, the last value provided *in the data package* will be used.

**substitute** (default: `True`): If `True`, overwrite existing values in the matrix; otherwise, add the given value to the existing value.

### Adding data interfaces

A data interface is a data source which can not be saved to a numpy array when the datapackage is created. There are many possible use cases for data interfaces, including:

* Data that is provided by an external source, such as a web service
* Data that comes from an infinite python generator
* Data from another programming language
* Data that needs processing steps before it can be directly inserted into a matrix

If an interface can return only one possible set of data, or if it can return an infinite generator (e.g. Monte Carlo uncertainty analysis), it should be a vector interface. If there are a given number of result data vectors, the interface should be an array interface.

#### Vector interfaces

Vector interfaces must be python objects that supports the [iterator interface](https://wiki.python.org/moin/Iterator), i.e. `__next__()`. Calling `next()` on the interface must return a Numpy vector.

#### Array interfaces

Array interfaces must support the following methods:

* `__getitem__(index)`: This is a magic method called when one uses `foo[bar]`. Should return a Numpy vector.

* `shape`: Used to determine the number of columns. Should return a 1-element tuple with the number of columns.

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
