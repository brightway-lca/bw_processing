# bw-processing

Library for storing numeric data for use in matrix-based calculations. Designed for use with the [Brightway life cycle assessment framework](https://brightway.dev/).

[![PyPI](https://img.shields.io/pypi/v/bw-processing.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/bw-processing.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/bw-processing)][pypi status]
[![License](https://img.shields.io/pypi/l/bw_processing)][license]

[![Read the documentation at https://bw-processing.readthedocs.io/](https://img.shields.io/readthedocs/bw-processing/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Testing](https://github.com/brightway-lca/bw_processing/actions/workflows/python-test.yml/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/brightway-lca/bw-processing/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/bw-processing/
[read the docs]: https://bw-processing.readthedocs.io/
[tests]: https://github.com/brightway-lca/bw_processing/actions/workflows/python-test.yml
[codecov]: https://app.codecov.io/gh/brightway-lca/bw-processing
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Table of Contents

- [Background](#background)
- [Concepts](#concepts)
  - [Data packages](#data-packages)
  - [Vectors versus arrays](#vectors-versus-arrays)
  - [Persistent versus dynamic](#persistent-versus-dynamic)
  - [Scale arrays](#scale-arrays)
  - [Parameter arrays for sensitivity analysis](#parameter-arrays-for-sensitivity-analysis)
  - [NaN as a sentinel value](#nan-as-a-sentinel-value)
  - [Policies](#policies)
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
* **Use [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) for file IO**. The use of this library allows for data packages to be stored on your local computer, or on [many logical or virtual file systems](https://docs.pyfilesystem.org/en/latest/guide.html).
* **Simpler handling of numeric values whose sign should be flipped**. Sometimes it is more convenient to specify positive numbers in dataset definitions, even though such numbers should be negative when inserted into the resulting matrices. For example, in the technosphere matrix in life cycle assessment, products produced are positive and products consumed are negative, though both values are given as positive in datasets. Brightway used to use a type mapping dictionary to indicate which values in a matrix should have their sign flipped after insertion. Such mapping dictionaries are brittle and inelegant. `bw_processing` uses an optional boolean vector, called `flip`, to indicate if any values should be flipped.
* **Per-exchange multiplicative scaling**. An optional float vector, called `scale`, can be attached to any resource group. Each element is a multiplicative factor applied to the corresponding data value — whether static or sampled stochastically — before it is inserted into the matrix. Typical uses are allocation factors and unit conversions. A value of `1.0` leaves the data unchanged.
* **Recording independent variables for sensitivity analysis**. An optional `params_array` can be attached to any resource group to record the values of model parameters (independent variables) that were used to generate the data. For array resources each column of `params_array` corresponds to the same column in `data_array`, making it straightforward to correlate inputs with outputs for sensitivity analysis methods such as Morris or Sobol.
* **Separation of uncertainty distribution parameters from other data**. Fitting data to a [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (PDF), or an estimate of such a PDF, is only one approach to quantitative uncertainty analysis. We would like to support other approaches, including [direct sampling from real data](https://github.com/PascalLesage/presamples/). Therefore, uncertainty distribution parameters are stored separately,  only loaded if needed, and are only one way to express quantitative uncertainty.

## Concepts

### Data packages

Data objects can be vectors or arrays. Vectors will always produce the same matrix, while arrays have multiple possible values for each element of the matrix. Arrays are a generalization of the [presamples library](https://github.com/PascalLesage/presamples/).

### Data needed for matrix construction

### Vectors versus arrays

Vectors and arrays differ in how many possible values they provide per matrix cell.

A **vector** provides one value per cell. Every time a vector resource is used, it produces the same result. This is the standard case for deterministic LCA calculations.

An **array** provides multiple possible values per cell, stored as columns of a 2-D numpy array. Each time the data package is iterated, a different column is selected and inserted into the matrix. This is used for:

* **Monte Carlo analysis** — columns hold independently sampled values drawn from the uncertainty distributions.
* **Scenario analysis** — each column represents a predefined scenario, such as a different technology mix or policy assumption.
* **Presamples** — a generalization of the [presamples library](https://github.com/PascalLesage/presamples/), where pre-drawn samples are stored for reproducibility.

Which column is selected on each iteration is controlled by the `sequential` and `combinatorial` policies; see [Policies](#policies).

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

#### Interface dehydrating and rehydrating

Serialized datapackages cannot contain executable code, both because of our chosen data formats, and for security reasons. Therefore, when loading a datapackage with an interface, that interface object needs to be reconstituted as Python code - we call this cycle dehydration and rehydration. Dehydration happens automatically when a datapackage is finalized with `finalize_serialization()`, but rehydration needs to be done manually using `rehydrate_interface()`. For example:

```python
from fsspec.implementations.zip import ZipFileSystem
from bw_processing import load_datapackage

my_dp = load_datapackage(ZipFileSystem("some-path.zip"))
my_dp.rehydrate_interface("some-resource-name", ExampleVectorInterface())
```

You can list the dehydrated interfaces present with `.dehydrated_interfaces()`.

You can store useful information for the interface object initialization under the resource key `config`. This can be used in instantiating an interface if you pass `initialize_with_config`:

```python
from fsspec.implementations.zip import ZipFileSystem
from bw_processing import load_datapackage
import requests
import numpy as np


class MyInterface:
    def __init__(self, url):
        self.url = url

    def __next__(self):
        return np.array(requests.get(self.url).json())


my_dp = load_datapackage(ZipFileSystem("some-path.zip"))
data_obj, resource_metadata = my_dp.get_resource("some-interface")
print(resource_metadata['config'])
>>> {"url": "example.com"}

my_dp.rehydrate_interface("some-interface", MyInterface, initialize_with_config=True)
# interface is substituted, need to retrieve it again
data_obj, resource_metadata = my_dp.get_resource("some-interface")
print(data_obj.url)
>>> "example.com"
```

### Scale arrays

Any resource group (persistent or dynamic, vector or array) can carry an optional `scale_array`: a one-dimensional float array of the same length as `indices_array`. Each element is a multiplicative factor applied to the corresponding data value before it is inserted into the matrix. The factor is applied to both static and stochastically-sampled values. A value of `1.0` leaves the data unchanged.

Typical use cases:

* **Allocation factors** — when a process produces multiple products, the exchange amounts must be partitioned between them. Storing the allocation coefficients as a `scale_array` keeps them alongside the data they modify without requiring a separate processing step.
* **Unit conversions** — when source data is expressed in a unit that differs from the matrix convention, a constant conversion factor can be stored as a `scale_array` rather than baked into every data value.

```python
import numpy as np
from bw_processing import create_datapackage
from bw_processing.constants import INDICES_DTYPE

dp = create_datapackage()
indices_array = np.array([(1, 4), (2, 5), (3, 6)], dtype=INDICES_DTYPE)
data_array = np.array([100.0, 200.0, 300.0])
scale_array = np.array([0.6, 1.0, 0.4])  # e.g. allocation factors

dp.add_persistent_vector(
    matrix="technosphere",
    name="my-process",
    indices_array=indices_array,
    data_array=data_array,
    scale_array=scale_array,
)
```

The stored resource has `kind="scale"` and can be retrieved via `dp.get_resource("my-process.scale")`. The `scale_array` must be a float dtype (`float32` or `float64`); passing an integer array raises `WrongDatatype`.

### Parameter arrays for sensitivity analysis

Any resource group can carry an optional `params_array` that records the values of the independent variables (model parameters) used to generate the data. This is the foundation for global sensitivity analysis workflows such as [Morris screening](https://en.wikipedia.org/wiki/Morris_method) or [Sobol indices](https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis).

**Shape conventions:**

| Resource type | `data_array` shape | `params_array` shape |
|---|---|---|
| persistent / dynamic vector | `(n_exchanges,)` | `(n_params,)` |
| persistent array | `(n_exchanges, n_scenarios)` | `(n_params, n_scenarios)` |
| dynamic array | `(n_exchanges, n_scenarios)` | `(n_params, n_scenarios)` — column count not validated against interface |

Column `j` of `params_array` describes the parameter configuration that produced column `j` of `data_array`. The `params_array` must be a float dtype.

#### Basic usage

```python
import numpy as np
from bw_processing import create_datapackage, INDICES_DTYPE

dp = create_datapackage()
indices = np.array([(1, 4), (2, 5)], dtype=INDICES_DTYPE)

# Store the parameter values alongside the data
dp.add_persistent_vector(
    matrix="technosphere",
    name="my-process",
    indices_array=indices,
    data_array=np.array([100.0, 200.0]),
    params_array=np.array([25.0, 1.013]),  # temperature (°C), pressure (atm)
)
```

#### Adding labels

`param_labels` is an optional list of label objects (strings or dicts) whose length must match `params_array.shape[0]`. When provided, a companion `name.param_labels.json` file is written inside the same resource group, containing a `"values"` list and an optional `"schema"` (a [JSON Schema](https://json-schema.org/) document). Pass `param_label_schema=StringLabelSchema()` for plain-string labels, or a `ParamLabelSchema` for structured dict labels.

```python
import numpy as np
from bw_processing import create_datapackage, INDICES_DTYPE, StringLabelSchema

dp = create_datapackage()
dp.add_persistent_vector(
    matrix="technosphere",
    name="my-process",
    indices_array=np.array([(1, 4)], dtype=INDICES_DTYPE),
    data_array=np.array([100.0]),
    params_array=np.array([25.0, 1.013]),
    param_labels=["temperature", "pressure"],
    param_label_schema=StringLabelSchema(),
)
```

#### Structured labels with a schema

Use `ParamLabelSchema` and `ParamLabelField` when labels are structured objects, and pass a `param_label_schema` to validate each label at write time:

```python
import numpy as np
from bw_processing import (
    create_datapackage,
    INDICES_DTYPE,
    ParamLabelField,
    ParamLabelSchema,
    StringLabelSchema,
)

dp = create_datapackage()

schema = ParamLabelSchema(
    fields=[
        ParamLabelField(name="name",      type="string"),
        ParamLabelField(name="database",  type="string"),
        ParamLabelField(name="year",      type="integer", required=False),
    ],
    description="Brightway activity reference",
)

dp.add_persistent_array(
    matrix="technosphere",
    name="sa-run",
    indices_array=np.array([(1, 4), (2, 5)], dtype=INDICES_DTYPE),
    data_array=np.array([[10.0, 20.0, 30.0],   # 2 exchanges × 3 scenarios
                         [40.0, 50.0, 60.0]]),
    params_array=np.array([[25.0, 30.0, 35.0],  # temperature: 2 params × 3 scenarios
                            [1.0,  1.1,  1.2]]),
    param_labels=[
        {"name": "electricity", "database": "ecoinvent", "year": 2020},
        {"name": "heat",        "database": "ecoinvent"},
    ],
    param_label_schema=schema,  # validates every label against the JSON Schema on write
)
```

`ParamLabelField.type` accepts the standard JSON Schema type names: `"string"`, `"integer"`, `"number"`, `"boolean"`. For plain-string labels you can also pass `param_label_schema=StringLabelSchema()` to make the schema explicit.

#### Retrieving params and labels

```python
params_data, _  = dp.get_resource("sa-run.params")       # numpy array
labels_data, _  = dp.get_resource("sa-run.param_labels")  # {"schema": ..., "values": [...]}

# Reconstruct the schema dataclass from the stored JSON Schema:
from bw_processing import schema_from_json_schema
schema = schema_from_json_schema(labels_data["schema"])
# → ParamLabelSchema(fields=[...])
```

#### Dependency

`params_array` validation uses the [`jsonschema`](https://python-jsonschema.readthedocs.io/) library, which is a required dependency of `bw_processing`.

### NaN as a sentinel value

A `NaN` value in a data vector or array is treated by `matrix_utils` as **"no data insertion"** — that element is skipped when the matrix is built or rebuilt, leaving the corresponding matrix cell at whatever value was written by an earlier package. This convention makes it straightforward to define scenario or override packages: set an element to `NaN` to inherit the base value, or to a real number to override it.

```python
import numpy as np
from bw_processing import create_datapackage, INDICES_DTYPE

# Base package — sets matrix[0,0]=5 and matrix[1,1]=7
dp_base = create_datapackage()
dp_base.add_persistent_vector(
    matrix="foo",
    name="base",
    indices_array=np.array([(0, 0), (1, 1)], dtype=INDICES_DTYPE),
    data_array=np.array([5.0, 7.0]),
)

# Scenario package — overrides matrix[1,1] but leaves matrix[0,0] untouched
dp_scenario = create_datapackage()
dp_scenario.add_persistent_vector(
    matrix="foo",
    name="scenario",
    indices_array=np.array([(0, 0), (1, 1)], dtype=INDICES_DTYPE),
    data_array=np.array([np.nan, 99.0]),
)
# When both packages are passed to MappedMatrix:
# matrix[0,0] == 5.0  (NaN in scenario → base value preserved)
# matrix[1,1] == 99.0 (non-NaN in scenario → override applied)
```

Note that `NaN` skipping is implemented in `matrix_utils`, not in `bw_processing` itself. `bw_processing` stores and retrieves the `NaN` values faithfully; the skip logic runs at matrix construction time.

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

Install using pip or conda (channel `conda-forge`). Depends on `numpy` and `pandas` (for reading and writing CSVs).

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

### Zip file compression

When writing a datapackage as a zip file via `generic_zipfile_filesystem`, compression is enabled by default (`zipfile.ZIP_DEFLATED`). Index arrays in particular compress extremely well (often to <10% of their original size) because they contain a limited set of repeated integer values. Float data arrays compress less, but the overall file size reduction is substantial.

```python
import zipfile
from bw_processing.io_helpers import generic_zipfile_filesystem

# Default: ZIP_DEFLATED (recommended — good compression, fast)
fs = generic_zipfile_filesystem(dirpath=some_path, filename="my.zip")

# Uncompressed — fastest write/read, largest file
fs = generic_zipfile_filesystem(dirpath=some_path, filename="my.zip", compression=zipfile.ZIP_STORED)

# LZMA — best compression ratio, but much slower to write
fs = generic_zipfile_filesystem(dirpath=some_path, filename="my.zip", compression=zipfile.ZIP_LZMA)
```

The `compression` and `compresslevel` arguments are passed directly to Python's `zipfile.ZipFile`. See the [Python docs](https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile) for all options.

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

<!-- github-only -->

[command-line reference]: https://bw-processing.readthedocs.io/en/latest/usage.html
[License]: https://github.com/brightway-lca/bw-processing/blob/main/LICENSE
[Contributor Guide]: https://github.com/brightway-lca/bw-processing/blob/main/CONTRIBUTING.md
[Issue Tracker]: https://github.com/brightway-lca/bw-processing/issues
