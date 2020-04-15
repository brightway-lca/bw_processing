# bw_processing

Tools to create structured arrays in a common format. Depends only on `numpy`.

[![Azure Status](https://dev.azure.com/mutel/Brightway%20CI/_apis/build/status/brightway-lca.bw_processing?branchName=master)](https://dev.azure.com/mutel/Brightway%20CI/_build/latest?definitionId=7&branchName=master) [![Travis Status](https://travis-ci.org/brightway-lca/bw_processing.svg?branch=master)](https://travis-ci.org/brightway-lca/bw_processing) [![Appveyor status](https://ci.appveyor.com/api/projects/status/ser0dd1au5jt409p?svg=true)](https://ci.appveyor.com/project/cmutel/bw-processing) [![Coverage Status](https://coveralls.io/repos/github/brightway-lca/bw_processing/badge.svg?branch=master)](https://coveralls.io/github/brightway-lca/bw_processing?branch=master)

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

Install using pip or conda (channel `cmutel`). Depends on `numpy`.

Has no explicit or implicit dependence on any other part of Brightway.

## Usage

## API

## Contributing

Your contribution is welcome! Please follow the [pull request workflow](https://guides.github.com/introduction/flow/), even for minor changes.

When contributing to this repository with a major change, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository.

Please note we have a [code of conduct](https://github.com/brightway-lca/bw_processing/blob/master/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

### Documentation and coding standards

* [Black formatting](https://black.readthedocs.io/en/stable/)
* [Sphinx docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)
* [Semantic versioning](http://semver.org/)

## Maintainers

* [Chris Mutel](https://github.com/cmutel/)

## License

[BSD-3-Clause](https://github.com/brightway-lca/bw_processing/blob/master/LICENSE). Copyright 2020 Chris Mutel.
