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

* **Consistent names for row and column fields**. Previously, these changed for each matrix, to reflect the role each row or column played in the model. Now they are always the same for all arrays, making the code simpler and easier to use.
* **Provision of metadata**. Numpy binary files are only data - `bw_processing` also produces a metadata file following the [data package standard](https://specs.frictionlessdata.io/data-package/). Things like data license, version, and unique id are now explicit and always included.
* **Simpler handling of values whose sign should be flipped**. Brightway used to use a type mapping dictionary to indicate which values in a matrix should have their sign flipped after insertion. For example, matrix values which represent the *consumption* of a good or service should be negative, while values that represent a *production* should be positive. The problem with a mapping dictionary with text keys is that it can be used or interpreted inconsistently. Instead, `bw_processing` simply has an additional column, `flip`.
* **Unified architecture for static and presamples data**. The way that static (i.e. only one possible value per input) and presamples (i.e. many possible values) data are handled is now exactly the same, allowing for simpler code and removing the possibility for inconsistent behaviour between these two.
* **Portability**. Processed arrays can include metadata that allows for reindexing on other machines, so that processed arrays can be distributed. Before, this was not possible, as integer IDs were randomly assigned, and would be different from machine to machine or even across Brightway projects. The same portability applies to presamples, making them more useful.
* **Dynamic data sources**. Data for matrix construction (both static and presamples) can be provided dynamically using a defined API. This allows for data sources on the cloud, or other external data interfaces.
* **In-memory data packages**. Data packages can be created in-memory and used for calculations.
* **Separation of uncertainty distribution parameters from other data**. Fitting data to an uncertainty distribution, or estimating such distributions, is only one approach to quantitative uncertainty analysis. We would like to support other approaches, including [direct sampling from real data](https://github.com/PascalLesage/presamples/). Therefore, uncertainty distribution parameters are stored separately, and only loaded if needed.

## Install

Install using pip or conda (channel `cmutel`). Depends on `numpy` and `pandas` (for CSV IO).

Has no explicit or implicit dependence on any other part of Brightway.

## Usage

### Basic concepts

Because this library supports multiple use cases, there are several dimensions to be aware of:

* In-memory versus on-disk. This is a (hopefully!) obvious difference. In-memory and on-disk presamples can be used together, but each datapackage must be only in-memory or on-disk.
* Static versus dynamic data: Static data can be completely loaded into memory and used directly or written to disk. Dynamic data is only resolved as the data is used (i.e. during matrix construction, not during package creation), and can be infinite generators.
* Creating versus loading. In-memory arrays can only be created and then used, but not loaded. On-disk arrays, on the other hand, must be created, stored, and then loaded to be used. This is because saving a package to disk flushes the data from memory.
* Vectors versus arrays. Each element in a vector can only have one possible value - this is the same as a static matrix. Arrays have the same number of rows as the length of a vector, but multiple possible values of each parameter. Each column is therefore one conssitent possible set of parameter values. See the [presample library for more information](https://github.com/PascalLesage/presamples/).
* Probability distributions. Probability distributions can only be defined for *static vectors*. Dynamic data can have uncertainty, but the dynamic data generator is responsible for modelling that uncertainty. Static arrays can't have uncertainty, as each column in the array represents the uncertainty and/or variability in the system performance or configuration.

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

### Adding data interfaces

A data interface is a data source which can not be saved to a numpy array when the datapackage is created. There are many possible use cases for data interfaces, including:

* Data that is provided by an external source, such as a web service
* Data that comes from an infinite python generator
* Data from another programming language
* Data that needs processing steps before it can be directly inserted into a matrix

Such data interfaces as supported by `bw_processing`, but the following conditions must be met:

#### Vector interfaces

* The interface itself must be a python object that supports the [iterator interface](https://wiki.python.org/moin/Iterator). Calling `next()` on the interface must return a numpy vector (1-d array).
* In addition to the interface, `indices_array` must be supplied. `indices_array` is a numpy structured array with the dtype `bw_processing.constants.INDICES_DTYPE`, and has the same length (and in the same order) as the vector returned by the interface.

You may also provide `flip_array`, a numpy structured array with the dtype `bw_processing.constants.FLIP_DTYPE`, which also has the same length and order as `indices_array`.

#### Array interfaces


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
