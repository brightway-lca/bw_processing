# bw_processing

Tools to create structured arrays in a common format. Depends only on `numpy`.

[![Azure Status](https://dev.azure.com/mutel/Brightway%20CI/_apis/build/status/brightway-lca.bw_processing?branchName=master)](https://dev.azure.com/mutel/Brightway%20CI/_build/latest?definitionId=7&branchName=master) [![Travis Status](https://travis-ci.org/brightway-lca/bw_processing.svg?branch=master)](https://travis-ci.org/brightway-lca/bw_processing) [![Coverage Status](https://coveralls.io/repos/github/brightway-lca/bw_processing/badge.svg?branch=master)](https://coveralls.io/github/brightway-lca/bw_processing?branch=master)

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
* **Simpler handling of negative values**. Brightway used to use a type mapping dictionary to indicate which values in a matrix should have their sign flipped after insertion. For example, matrix values which represent the *consumption* of a good or service should be negative, while values that represent a *production* should be positive. The problem with a mapping dictionary with text keys is that it can be used or interpreted inconsistently. Instead, `bw_processing` has an additional column: `flip`.

## Install

Install using pip or conda (channel `cmutel`). Depends on `numpy` and `pandas` (for CSV IO).

Has no explicit or implicit dependence on any other part of Brightway.

## Usage

Let's see it in action:

```python
In [1]: from bw_processing import create_calculation_package

In [2]: create_calculation_package(
   ...:     name="foo",
   ...:     resources=[{
   ...:         "name": "data from a calculation",
   ...:         "path": "bar",
   ...:         "matrix": "some-matrix-label",
   ...:         "data": [
   ...:             {"row": 1, "col": 2, "amount": 3},
   ...:             {"row": 14, "col": 15, "amount": 16},
   ...:         ],
   ...:     }],
   ...:     id_="some-id",
   ...:     metadata={"some": "stuff"}
   ...: )
Out[2]:
{
    'bar.npy': array(
        [
            ( 1,  2, 2147483647, 2147483647, 0,  3.,  3., nan, nan, nan, nan, False, False),
            (14, 15, 2147483647, 2147483647, 0, 16., 16., nan, nan, nan, nan, False, False)
        ],
        dtype=[
            ('row_value', '<u4'), ('col_value', '<u4'), ('row_index', '<u4'),
            ('col_index', '<u4'), ('uncertainty_type', 'u1'), ('amount', '<f4'),
            ('loc', '<f4'), ('scale', '<f4'), ('shape', '<f4'), ('minimum', '<f4'),
            ('maximum', '<f4'), ('negative', '?'), ('flip', '?')
        ]
    ),
    'datapackage': {
        'profile': 'data-package',
        'name': 'foo',
        'id': 'some-id',
        'licenses': [{
            'name': 'ODC-PDDL-1.0',
            'path': 'http://opendatacommons.org/licenses/pddl/',
            'title': 'Open Data Commons Public Domain Dedication and License v1.0'
        }],
        'resources': [{
            'format': 'npy',
            'mediatype': 'application/octet-stream',
            'path': 'bar.npy',
            'name': 'data from a calculation',
            'profile': 'data-resource',
            'matrix': 'some-matrix-label'
        }],
        'some': 'stuff',
        'created': '2020-04-15T21:01:35.508622Z'
    }
}
```

In this example, we returned the processed data package as a dictionary in memory, but normally `bw_processing` is used to persist data to disk.

There is also a utility function [`load_calculation_package`](#loading), which loads a saved calculation package in the same format as was returned by the example.

### `create_calculation_package`

`create_calculation_package` has the following input arguments:

* *name*: The name of this package. It is stored in the data package metadata.
* *resources*: A list of data resources. See [data resources](#data-resources).
* *path* (optional): Where the package should be saved. If `None` (default), it is not saved to disk. Otherwise, `path` can be a `str` or a `pathlib.Path`. It doesn't have to exist yet.
* *id_* (optional): The unique id of this package. This field is required by the data package standard. Generated automatically if not provided.
* *metadata* (optional): Any additional metadata for the package (metadata for individual resources is given in the resource definition above).
* *replace* (optional). Replace existing package if already present. Default is `True`.
* *compress* (optional). Save package as a single zip archive instead of a directory. Default is `True`.

### Data resources

Input data can be provided in a number of ways. It can already be in memory, e.g. a numpy array; it can be generated on demand, e.g. from a generator; or it can be the result of a more complicated function. In any case, the data object

### Outputs



## API

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
