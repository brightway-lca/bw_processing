# Changelog

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
