# `bw_processing` Changelog

# [1.0] - 2024-08-19

* Move default indices type to 64 bit integers

### [0.9.6] - 2024-07-02

* [PR #68: replace `np.NaN` by `np.nan`](https://github.com/brightway-lca/bw_processing/pull/68)

### [0.9.5] - 2024-05-29

* [PR #66: Change from `pyfilesystem2` to `fsspec`](https://github.com/brightway-lca/bw_processing/issues/65)
* Changed packaging

### [0.9.3] - 2024-05-12

* Fix compatibility function for datetime utcnow for older Python versions

### [0.9.2] - 2024-05-11

* Add compatibility function for datetime utcnow for older Python versions

### [0.9.1] - 2024-05-10

* Fix deprecation warning for datetime utcnow

## [0.9] - 2023-09-18

* Add support for parquet files

### [0.8.5] - 2023-08-24

* Switch versioning scheme to help WASM packaging

### [0.8.4] - 2023-08-23

* Packaging fixes

### [0.8.3] - 2022-09-15

* Change distribution name to fix Windows installation issues (`bw-processing` instead of `bw_processing`)
* Change packaging to [our common standard](https://github.com/brightway-lca/cookiecutter-brightwaylib)

### [0.8.2] - 2022-09-15

* Don't export DataFrame index columns

### [0.8.1] - 2022-09-01

* Add `simple_graph` convenience function

## [0.8] - 2022-04-11

* Add `merge_datapacakges_with_mask` function

### [0.7.1] - 2021-11-26

* Filtered data packages now have the same `.fs` attribute as their parent package

## [0.7] - 2021-11-24

* **Backwards incompatible**: `Datapackage.groups` no longer sorts group labels. Sorting breaks implied ordering for matrix construction.

### [0.6.1] - 2021-11-24

* Moved dehydration and rehydration to `DatapackageBase` so they are available to `FilteredDatapackage` as well.

## [0.6] - 2021-11-24

* Define, document, and test interface dehydration and rehydration. `Datapackage.define_interface_resource` changed to `Datapackage.rehydrate_interface`.
* Removed the cache from `Datapackage.get_resource`. It was counterproductive.

## [0.5] - 2021-11-03

* Add tests for resource dtypes and shapes in add_X functions. Fixes [#11](https://github.com/brightway-lca/bw_processing/issues/11).
* Combined  `exclude_resource_group` and `exclude_resource_group_kind` into `exclude`

## [0.4] - 2021-10-04

* Fix [#6 - `del_resource` should support resource groups](https://github.com/brightway-lca/bw_processing/issues/6)
* Fix [#7 - Error out when adding existing resource to package](https://github.com/brightway-lca/bw_processing/issues/7)
* Fix [#9 - Multiple readings of same proxied data resource in FilteredDataPackage causes errors](https://github.com/brightway-lca/bw_processing/issues/9)
* Fix [#10 - Add function for filtering resource groups](https://github.com/brightway-lca/bw_processing/issues/10)

## [0.3.1] - 2021-06-02

* Keep package indexers when creating a `FilteredDatapackage`.

## [0.3] - 2021-05-19

Nearly complete reconceptualization of the package structure and logic, based around the use of [PyFilesystem2](https://docs.pyfilesystem.org/en/latest/). Much more complete package, with better testing and documentation.

### [0.1.2] - 2020-04-15

Small background improvements to package creation

### [0.1.1] - 2020-01-22

Remove `bw_projects` as a dependency

## [0.1] - 2019-11-12

First public release
