# bw_processing

Tools to create structured arrays in a common format. Depends only on `numpy`.

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

The [Brightway LCA framework](https://brightway.dev/) has stored data used in constructing matrices in binary form as numpy arrays for years. This package is an evolution of that idea, and adds the following features:

* **Consistent names for row and column fields**. Instead of having these vary by the kind of matrix to be constructed, they are always the same, making the code simpler and easier to use.
* **Provision of metadata**. Numpy binary files are only data - `bw_processing` also produces a metadata file following the [data package standard](https://specs.frictionlessdata.io/data-package/). Things like data license, version, and unique id are now explicit and always included.
* **Simpler handling of values whose sign should be flipped**. Brightway used to use a type mapping dictionary to indicate which values in a matrix should have their sign flipped after insertion. For example, matrix values which represent the *consumption* of a good or service should be negative, while values that represent a *production* should be positive. The problem with a mapping dictionary with text keys is that it can be used or interpreted inconsistently. Instead, `bw_processing` has an additional column: `flip`.
* **Unified architecture for static and presamples data**. Both the processed data arrays, and presamples (both intermediate results and final matrix substitution data) share the same data and metadata format.
* **Portability**. Processed arrays can include metadata that allows for reindexing on other machines, so that processed arrays can be distributed. Before, this was not possible, as integer IDs were randomly assigned, and would be different from machine to machine or even across Brightway projects. The same portability applies to presamples.
* **Allow for data provision by cloud or external data interfaces**. Both structured arrays and presamples can be defined by external or dynamic data sources. Note that the existing functionality is relatively basic, and will be expanded upon in future releases.

## Install

Install using pip or conda (channel `cmutel`). Depends on `numpy` and `pandas` (for CSV IO).

Has no explicit or implicit dependence on any other part of Brightway.

## Usage

The main interface for using this library is the `Datapackage` class, which is instantiated with either `Datapackage.create()` or `Datapackage.load()`. Data packages can be stored either in-memory, in a directory, or as a zipfile.

Each data package has some generic metadata (i.e. when it was created, the data license), metadata specific to each data resource (how it can be used in calculations, its relationship to other data resources), and the actual data resources themselves.

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
