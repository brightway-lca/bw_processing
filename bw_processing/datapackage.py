from .constants import DEFAULT_LICENSES
from .errors import (
    Closed,
    LengthMismatch,
    NonUnique,
    InvalidMimetype,
    PotentialInconsistency,
)
from .filesystem import clean_datapackage_name
from .io_helpers import file_writer, file_reader
from .proxies import UndefinedInterface, Proxy
from .utils import (
    check_name,
    check_suffix,
    load_bytes,
    resolve_dict_iterator,
    NoneSorter,
)
from fs.base import FS
from fs.errors import ResourceNotFound
from fs.memoryfs import MemoryFS
from functools import partial
from typing import Union, Any
import datetime
import numpy as np
import pandas as pd
import uuid


class DatapackageBase:
    """Base class for datapackages. Not for normal use - you should use either `Datapackage` or `FilteredDatapackage`."""

    def __init__(self):
        self._finalized = False
        self._cache = {}
        self._modified = set()

    def __get_resources(self) -> list:
        return self.metadata["resources"]

    def __set_resources(self, dct: dict) -> None:
        self.metadata["resources"] = dct

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return key in {o["name"] for o in self.resources} or key in self.groups

    resources = property(__get_resources, __set_resources)

    @property
    def groups(self) -> dict:
        return {
            label: self.filter_by_attribute("group", label)
            for label in sorted(
                {x.get("group") for x in self.resources},
                key=lambda x: NoneSorter() if x is None else x,
            )
        }

    def _get_index(self, name_or_index: Union[str, int]) -> int:
        """Get index of a resource by name or index.

        Returning the same number is a bit silly, but makes the other code simpler :)

        Raises:

            * IndexError: ``name_or_index`` was too big
            * ValueError: Name ``name_or_index`` not found
            * NonUnique: Name ``name_or_index`` not unique in given resources

        """
        if isinstance(name_or_index, int):
            if name_or_index >= len(self.resources):
                raise IndexError(
                    "Index {} given, but only {} resources available".format(
                        name_or_index, len(self.resources)
                    )
                )
            return name_or_index
        else:
            indices = []
            for i, o in enumerate(self.resources):
                if o["name"] == name_or_index:
                    indices.append(i)

            if not indices:
                raise KeyError("Name {} not found in metadata".format(name_or_index))
            elif len(indices) > 1:
                raise NonUnique("This name present at indices: {}".format(indices))
            else:
                return indices[0]

    def del_resource(self, name_or_index: Union[str, int]) -> None:
        """Remove a resource, and delete its data file, if any."""
        if self._modified:
            raise PotentialInconsistency(
                "Datapackage is modified; save modifications or reload"
            )

        index = self._get_index(name_or_index)

        try:
            self.fs.remove(self.resources[index]["path"])
        except (KeyError, ResourceNotFound):
            # Interface has no path
            pass

        del self.resources[index]
        del self.data[index]

    def del_resource_group(self, name: str) -> None:
        """Remove a resource group, and delete its data files, if any."""
        if self._modified:
            raise PotentialInconsistency(
                "Datapackage is modified; save modifications or reload"
            )

        indices = [
            i
            for i, resource in enumerate(self.resources)
            if resource.get("group") == name
        ]

        for obj in (obj for i, obj in enumerate(self.resources) if i in indices):
            try:
                self.fs.remove(obj["path"])
            except (KeyError, ResourceNotFound):
                # Interface has no path
                pass

        self.resources = [
            obj for i, obj in enumerate(self.resources) if i not in indices
        ]
        self.data = [obj for i, obj in enumerate(self.data) if i not in indices]

    def get_resource(self, name_or_index: Union[str, int]) -> (Any, dict):
        """Return data and metadata for ``name_or_index``.

        Args:

            * name_or_index: Name (str) or index (int) of a resource in the existing metadata.

        Raises:

            * IndexError: Integer index out of range of given metadata
            * ValueError: String name not present in metadata
            * NonUnique: String name present in two resource metadata sections

        Returns:

            (data object, metadata dict)

        """
        try:
            self._cache[name_or_index]
        except KeyError:
            index = self._get_index(name_or_index)

            if isinstance(self.data[index], (Proxy, partial)):
                self.data[index] = self.data[index]()

            self._cache[name_or_index] = (self.data[index], self.resources[index])
        return self._cache[name_or_index]

    def filter_by_attribute(self, key: str, value: Any) -> "FilteredDatapackage":
        """Create a new ``FilteredDatapackage`` which satisfies the filter ``resource[key] == value``.

        All included objects are the same as in the original data package, i.e. no copies are made. No checks are made to ensure consistency with modifications to the original datapackage after the creation of this filtered datapackage.

        This method was introduced to allow for the efficient construction of matrices; each datapackage can have data for multiple matrices, and we can then create filtered datapackages which exclusively have data for the matrix of interest. As such, they should be considered read-only, though this is not enforced."""
        fdp = FilteredDatapackage()
        fdp.metadata = {k: v for k, v in self.metadata.items() if k != "resources"}
        fdp.metadata["resources"] = []
        to_include = [
            i for i, resource in enumerate(self.resources) if resource.get(key) == value
        ]
        fdp.data = [o for i, o in enumerate(self.data) if i in to_include]
        fdp.resources = [o for i, o in enumerate(self.resources) if i in to_include]
        if hasattr(self, "indexer"):
            fdp.indexer = self.indexer
        return fdp

    def exclude_resource_group(self, group: str) -> "FilteredDatapackage":
        """Filter a datapackage to exclude a resource group by group name.

        Use ``filter_by_attribute("group", "group_name")`` to get only this resource group.

        """
        fdp = FilteredDatapackage()
        fdp.metadata = {k: v for k, v in self.metadata.items() if k != "resources"}
        fdp.metadata["resources"] = []

        if hasattr(self, "indexer"):
            fdp.indexer = self.indexer

        to_include = [
            i
            for i, resource in enumerate(self.resources)
            if resource.get("group") != group
        ]
        fdp.data = [o for i, o in enumerate(self.data) if i in to_include]
        fdp.resources = [o for i, o in enumerate(self.resources) if i in to_include]
        return fdp

    def exclude_resource_group_kind(
        self, group: str, kind: str
    ) -> "FilteredDatapackage":
        """Filter a datapackage to exclude a certain kinds of resources in a given resource group.

        The intended use case is to do complicated calculation setups, like distributions from one resource group but just the static array from a different resource group in the same datapackage.

        ``group`` and ``kind`` are the group and kind labels of the resource to exclude.

        """
        fdp = FilteredDatapackage()
        fdp.metadata = {k: v for k, v in self.metadata.items() if k != "resources"}
        fdp.metadata["resources"] = []

        if hasattr(self, "indexer"):
            fdp.indexer = self.indexer

        to_exclude = [
            i
            for i, resource in enumerate(self.resources)
            if resource.get("group") == group and resource.get("kind") == kind
        ]
        fdp.data = [o for i, o in enumerate(self.data) if i not in to_exclude]
        fdp.resources = [o for i, o in enumerate(self.resources) if i not in to_exclude]
        return fdp


class FilteredDatapackage(DatapackageBase):
    """A subset of a datapackage. Used in matrix construction or other data manipulation operations.

    Should be treated as read-only."""

    pass


class Datapackage(DatapackageBase):
    """
    Interface for creating, loading, and using numerical datapackages for Brightway.

    Note that there are two entry points to using this class, both separate functions: ``create_datapackage()`` and ``load_datapackage()``. Do not create an instance of the class with ``Datapackage()``, unless you like playing with danger :)

    Data packages can be stored in memory, in a directory, or in a zip file. When creating data packages for use later, don't forget to call ``.finalize_serialization()``, or the metadata won't be written and the data package won't be usable.

    Potential gotchas:

    * There is currently no way to modify a zipped data package once it is finalized.
    * Resources that are interfaces to external data sources (either in Python or other) can't be saved, but must be recreated each time a data package is used.

    """

    # To allow these packages to be used as Python keys
    def __hash__(self):
        return hash((self.fs, self.metadata))

    def __eq__(self, other):
        return (self.fs, self.metadata) == (other.fs, other.metadata)

    def _check_length_consistency(self) -> None:
        if len(self.resources) != len(self.data):
            raise LengthMismatch(
                "Number of resources ({}) doesn't match number of data objects ({})".format(
                    len(self.resources), len(self.data)
                )
            )

    def _load(
        self, fs: FS, mmap_mode: Union[None, str] = None, proxy: bool = False
    ) -> None:
        self.fs = fs
        self.metadata = file_reader(
            fs=self.fs, resource="datapackage.json", mimetype="application/json"
        )
        self.data = []
        self._load_all(mmap_mode=mmap_mode, proxy=proxy)

    def _load_all(
        self, mmap_mode: Union[None, str] = None, proxy: bool = False
    ) -> None:
        for resource in self.resources:
            try:
                self.data.append(
                    file_reader(
                        fs=self.fs,
                        resource=resource["path"],
                        mimetype=resource["mediatype"],
                        proxy=proxy,
                        mmap_mode=mmap_mode,
                    )
                )
            except (InvalidMimetype, KeyError):
                self.data.append(UndefinedInterface())

    def _create(
        self,
        fs: Union[FS, None],
        name: Union[str, None],
        id_: Union[str, None],
        metadata: Union[dict, None],
        combinatorial: bool = False,
        sequential: bool = False,
        seed: Union[int, None] = None,
        sum_intra_duplicates: bool = True,
        sum_inter_duplicates: bool = False,
    ) -> None:
        """Start a new data package.

        All metadata elements should follow the `datapackage specification <https://frictionlessdata.io/specs/data-package/>`__.

        Licenses are specified as a list in ``metadata``. The default license is the `Open Data Commons Public Domain Dedication and License v1.0 <http://opendatacommons.org/licenses/pddl/>`__.
        """
        name = clean_datapackage_name(name or uuid.uuid4().hex)
        check_name(name)

        self.fs = fs or MemoryFS()

        self.metadata = {
            "profile": "data-package",
            "name": name,
            "id": id_ or uuid.uuid4().hex,
            "licenses": (metadata or {}).get("licenses", DEFAULT_LICENSES),
            "resources": [],
            "created": datetime.datetime.utcnow().isoformat("T") + "Z",
            "combinatorial": combinatorial,
            "sequential": sequential,
            "seed": seed,
            "sum_intra_duplicates": sum_intra_duplicates,
            "sum_inter_duplicates": sum_inter_duplicates,
        }
        for k, v in (metadata or {}).items():
            if k not in self.metadata:
                self.metadata[k] = v

        self.data = []

    def _substitute_interfaces(self) -> None:
        """Substitute an interface resource with ``UndefinedInterface``, in preparation for finalizing data on disk."""
        interface_indices = [
            index
            for index, obj in enumerate(self.resources)
            if obj["profile"] == "interface"
        ]

        for index in interface_indices:
            self.data[index] = UndefinedInterface()

    def finalize_serialization(self) -> None:
        if self._finalized:
            raise Closed("Datapackage already finalized")
        elif isinstance(self.fs, MemoryFS):
            raise ValueError("In-memory file systems can't be serialized")

        self._substitute_interfaces()
        self._check_length_consistency()

        file_writer(
            data=self.metadata,
            fs=self.fs,
            resource="datapackage.json",
            mimetype="application/json",
        )
        self.fs.close()

    def define_interface_resource(
        self, name_or_index: Union[str, int], resource: Any
    ) -> None:
        """Substitute the undefined interface with ``resource``"""
        self.data[self._get_index(name_or_index)] = resource

    def _prepare_modifications(self) -> None:
        self._check_length_consistency()

        if self._finalized:
            raise Closed("Datapackage already finalized")

    def _prepare_name(self, name: str) -> str:
        name = name or uuid.uuid4().hex

        existing_names = {o["name"] for o in self.resources}
        existing_groups = {o["group"] for o in self.resources if o.get("group")}
        if name in existing_names:
            raise NonUnique("This name already used")
        if name in existing_groups:
            raise NonUnique("This group name already used")

        return name

    def add_persistent_vector_from_iterator(
        self,
        *,  # Forces use of keyword arguments
        matrix: str = None,
        name: Union[str, None] = None,
        dict_iterator: Any = None,
        nrows: Union[int, None] = None,
        **kwargs,
    ) -> None:
        """Create a persistant vector from an iterator. Uses the utility function ``resolve_dict_iterator``.

        This is the **only array creation method which produces sorted arrays**."""
        name = self._prepare_name(name)
        (
            data_array,
            indices_array,
            distributions_array,
            flip_array,
        ) = resolve_dict_iterator(dict_iterator, nrows)
        self.add_persistent_vector(
            matrix=matrix,
            name=name,
            nrows=len(data_array),
            data_array=data_array,
            indices_array=indices_array,
            flip_array=flip_array,
            distributions_array=distributions_array,
            **kwargs,
        )

    def add_persistent_vector(
        self,
        *,  # Forces use of keyword arguments
        matrix: str,
        indices_array: np.ndarray,
        name: Union[str, None] = None,
        data_array: Union[np.ndarray, None] = None,
        flip_array: Union[np.ndarray, None] = None,
        distributions_array: Union[np.ndarray, None] = None,
        keep_proxy: bool = False,
        **kwargs,
    ) -> None:
        """ """
        self._prepare_modifications()

        # Check lengths

        kwargs.update(
            {"matrix": matrix, "category": "vector", "nrows": len(indices_array)}
        )
        name = self._prepare_name(name)

        self._add_numpy_array_resource(
            array=load_bytes(indices_array),
            name=name + ".indices",
            group=name,
            kind="indices",
            keep_proxy=keep_proxy,
            **kwargs,
        )
        if data_array is not None:
            self._add_numpy_array_resource(
                array=load_bytes(data_array),
                group=name,
                name=name + ".data",
                kind="data",
                keep_proxy=keep_proxy,
                **kwargs,
            )
        if distributions_array is not None:
            # If no uncertainty, don't need to store it
            if (distributions_array["uncertainty_type"] < 2).sum() < len(
                distributions_array
            ):
                self._add_numpy_array_resource(
                    array=load_bytes(distributions_array),
                    name=name + ".distributions",
                    group=name,
                    kind="distributions",
                    keep_proxy=keep_proxy,
                    **kwargs,
                )
        if flip_array is not None and flip_array.sum():
            self._add_numpy_array_resource(
                array=load_bytes(flip_array),
                group=name,
                name=name + ".flip",
                kind="flip",
                keep_proxy=keep_proxy,
                **kwargs,
            )

    def add_persistent_array(
        self,
        *,  # Forces use of keyword arguments
        matrix: str,
        data_array: np.ndarray,
        indices_array: np.ndarray,
        name: Union[str, None] = None,
        flip_array: Union[None, np.ndarray] = None,
        keep_proxy: bool = False,
        **kwargs,
    ) -> None:
        """ """
        self._prepare_modifications()

        kwargs.update(
            {"matrix": matrix, "category": "array", "nrows": len(indices_array)}
        )
        name = self._prepare_name(name)

        self._add_numpy_array_resource(
            array=load_bytes(data_array),
            name=name + ".data",
            group=name,
            kind="data",
            keep_proxy=keep_proxy,
            **kwargs,
        )
        self._add_numpy_array_resource(
            array=load_bytes(indices_array),
            name=name + ".indices",
            kind="indices",
            group=name,
            keep_proxy=keep_proxy,
            **kwargs,
        )
        if flip_array is not None and flip_array.sum():
            self._add_numpy_array_resource(
                array=load_bytes(flip_array),
                group=name,
                name=name + ".flip",
                kind="flip",
                keep_proxy=keep_proxy,
                **kwargs,
            )

    def write_modified(self):
        """Write the data in modified files to the filesystem (if allowed)."""
        for index in self._modified:
            file_writer(
                data=self.data[index],
                fs=self.fs,
                resource=self.resources[index]["path"],
                mimetype=self.resources[index]["mediatype"],
            )

        self._modified = set()

    def _add_numpy_array_resource(
        self,
        *,
        array: np.ndarray,
        name: str,
        matrix: str,
        kind: str,
        keep_proxy: bool = False,
        **kwargs,
    ) -> None:
        filename = check_suffix(name, ".npy")

        if not isinstance(self.fs, MemoryFS):
            file_writer(
                data=array,
                fs=self.fs,
                resource=filename,
                mimetype="application/octet-stream",
            )

        if keep_proxy:
            self.data.append(
                file_reader(
                    fs=self.fs,
                    resource=filename,
                    mimetype="application/octet-stream",
                    proxy=True,
                    **kwargs,
                )
            )
        else:
            self.data.append(array)

        resource = {
            # Datapackage generic
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": name,
            # Brightway specific
            "matrix": matrix,
            "kind": kind,
            "path": str(filename),
        }
        resource.update(**kwargs)
        self.resources.append(resource)

    def add_dynamic_vector(
        self,
        *,
        matrix: str,
        interface: Any,
        indices_array: np.ndarray,  # Not interface
        name: Union[str, None] = None,
        flip_array: Union[None, np.ndarray] = None,  # Not interface
        keep_proxy: bool = False,
        **kwargs,
    ) -> None:
        self._prepare_modifications()

        kwargs.update(
            {"matrix": matrix, "category": "vector", "nrows": len(indices_array)}
        )
        name = self._prepare_name(name)

        # Do something with dynamic vector

        self._add_numpy_array_resource(
            array=load_bytes(indices_array),
            name=name + ".indices",
            group=name,
            kind="indices",
            keep_proxy=keep_proxy,
            **kwargs,
        )
        if flip_array is not None and flip_array.sum():
            self._add_numpy_array_resource(
                array=load_bytes(flip_array),
                group=name,
                name=name + ".flip",
                kind="flip",
                keep_proxy=keep_proxy,
                **kwargs,
            )

        self.data.append(interface)
        resource = {
            "profile": "interface",
            "name": name + ".data",
            "group": name,
            "kind": "data",
        }
        resource.update(**kwargs)
        self.resources.append(resource)

    def add_dynamic_array(
        self,
        *,
        matrix: str,
        interface: Any,
        indices_array: np.ndarray,  # Not interface
        name: Union[str, None] = None,
        flip_array: Union[None, np.ndarray] = None,
        keep_proxy: bool = False,
        **kwargs,
    ) -> None:
        """`interface` must support the presamples API."""
        self._prepare_modifications()

        if isinstance(flip_array, np.ndarray) and not flip_array.sum():
            flip_array = None

        kwargs.update(
            {"matrix": matrix, "category": "array", "nrows": len(indices_array)}
        )
        name = self._prepare_name(name)

        self._add_numpy_array_resource(
            array=load_bytes(indices_array),
            name=name + ".indices",
            group=name,
            kind="indices",
            keep_proxy=keep_proxy,
            **kwargs,
        )
        if flip_array is not None:
            self._add_numpy_array_resource(
                array=load_bytes(flip_array),
                group=name,
                name=name + ".flip",
                kind="flip",
                keep_proxy=keep_proxy,
                **kwargs,
            )

        self.data.append(interface)
        resource = {
            "profile": "interface",
            "name": name + ".data",
            "group": name,
            "kind": "data",
        }
        resource.update(**kwargs)
        self.resources.append(resource)

    def add_csv_metadata(
        self, *, dataframe: pd.DataFrame, valid_for: list, name: str = None, **kwargs
    ) -> None:
        """Add an iterable metadata object to be stored as a CSV file.

        The purpose of storing metadata is to enable data exchange; therefore, this method assumes that data is written to disk.

        The normal use case of this method is to link integer indices from either structured or presample arrays to a set of fields that uniquely identifies each object. This allows for matching based on object attributes from computer to computer, where database ids or other computer-generated codes might not be consistent.

        Uses pandas to store and load data; therefore, metadata must already be a pandas dataframe.

        In contrast with presamples arrays, ``iterable_data_source`` cannot be an infinite generator. We need a finite set of data to build a matrix.

        In contrast to ``self.create_structured_array``, this always stores the dataframe in ``self.data``; no proxies are used.

        Args:

            * dataframe: Dataframe to be persisted to disk.
            * valid_for: List of resource names that this metadata is valid for; must be either structured or presample indices arrays. Each item in ``valid_for`` has the form ``("resource_name", "rows" or "cols")``. ``resource_name`` should be either a structured or a presamples indices array.
            * name (optional): The name of this resource. Names must be unique in a given data package
            * extra (optional): Dict of extra metadata

        Returns:

            Nothing, but appends objects to ``self.metadata['resources']`` and ``self.data``.

        Raises:

            * AssertionError: If inputs are not in correct form
            * AssertionError: If ``valid_for`` refers to unavailable resources

        """
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(valid_for, list)

        assert all(x in self.groups for x, y in valid_for)

        name = self._prepare_name(name)
        self._prepare_modifications()

        filename = check_suffix(name, ".csv")

        file_writer(data=dataframe, fs=self.fs, resource=filename, mimetype="text/csv")
        self.data.append(dataframe)

        kwargs.update(
            {
                # Datapackage generic
                "profile": "data-resource",
                "mediatype": "text/csv",
                "path": filename,
                "name": name,
                # Brightway specific
                "valid_for": valid_for,
            }
        )
        self.resources.append(kwargs)

    def add_json_metadata(
        self, *, data: Any, valid_for: str, name: str = None, **kwargs
    ) -> None:
        """Add an iterable metadata object to be stored as a JSON file.

        The purpose of storing metadata is to enable data exchange; therefore, this method assumes that data is written to disk.

        The normal use case of this method is to provide names and other metadata for parameters whose values are stored as presamples arrays. The length of ``data`` should match the number of rows in the corresponding presamples array, and ``data`` is just a list of string labels for the parameters. However, this method can also be used to store other metadata, e.g. for external data resources.

        In contrast to ``self.create_structured_array``, this always stores the dataframe in ``self.data``; no proxies are used.

        Args:

            * data: Data to be persisted to disk.
            * valid_for: Name of structured data or presample array that this metadata is valid for.
            * name (optional): The name of this resource. Names must be unique in a given data package
            * extra (optional): Dict of extra metadata

        Returns:

            Nothing, but appends objects to ``self.metadata['resources']`` and ``self.data``.

        Raises:

            * AssertionError: If inputs are not in correct form
            * AssertionError: If ``valid_for`` refers to unavailable resources

        """
        assert isinstance(valid_for, str)
        assert valid_for in self.groups

        self._prepare_modifications()

        name = name or uuid.uuid4().hex
        check_name(name)

        filename = check_suffix(name, ".json")

        file_writer(
            data=data, fs=self.fs, resource=filename, mimetype="application/json"
        )
        self.data.append(data)

        kwargs.update(
            {
                # Datapackage generic
                "profile": "data-resource",
                "mediatype": "application/json",
                "path": str(filename),
                "name": name,
                # Brightway specific
                "valid_for": valid_for,
            }
        )
        self.resources.append(kwargs)


def create_datapackage(
    fs: Union[FS, None] = None,
    name: Union[None, str] = None,
    id_: Union[None, str] = None,
    metadata: Union[dict, None] = None,
    combinatorial: bool = False,
    sequential: bool = False,
    seed: Union[int, None] = None,
    sum_intra_duplicates: bool = True,
    sum_inter_duplicates: bool = False,
) -> Datapackage:
    """Create a new data package.

    All arguments are optional; if a `PyFilesystem2 <https://docs.pyfilesystem.org/en/latest/>`__ filesystem is not provided, a `MemoryFS <https://docs.pyfilesystem.org/en/latest/reference/memoryfs.html>`__ will be used.

    All metadata elements should follow the `datapackage specification <https://frictionlessdata.io/specs/data-package/>`__.

    Licenses are specified as a list in ``metadata``. The default license is the `Open Data Commons Public Domain Dedication and License v1.0 <http://opendatacommons.org/licenses/pddl/>`__.

    Args:

        * fs: A `Filesystem`, optional. A new `MemoryFS` is used if not provided.
        * name: `str`, optional. A new uuid is used if not provided.
        * id_. `str`, optional. A new uuid is used if not provided.
        * metadata. `dict`, optional. Metadata dictionary following datapackage specification; see above.
        * combinatorial. `bool`, default `False.: Policy on how to sample columns across multiple data arrays; see readme.
        * sequential. `bool`, default `False.: Policy on how to sample columns in data arrays; see readme.
        * seed. `int`, optional. Seed to use in random number generator.
        * sum_intra_duplicates. `bool`, default `True`. Should duplicate elements in a single data resource be summed together, or should the last value replace previous values.
        * sum_inter_duplicates. `bool`, default `False`. Should duplicate elements in across data resources be summed together, or should the last value replace previous values. Order of data resources is given by the order they are added to the data package.

    Returns:

        A `Datapackage` instance.

    """
    obj = Datapackage()
    obj._create(
        fs=fs,
        name=name,
        id_=id_,
        metadata=metadata,
        sequential=sequential,
        combinatorial=combinatorial,
        seed=seed,
        sum_intra_duplicates=sum_intra_duplicates,
        sum_inter_duplicates=sum_inter_duplicates,
    )
    return obj


def load_datapackage(
    fs_or_obj: Union[DatapackageBase, FS],
    mmap_mode: Union[None, str] = None,
    proxy: bool = False,
) -> Datapackage:
    """Load an existing datapackage.

    Can load proxies to data instead of the data itself, which can be useful when interacting with large arrays or large packages where only a subset of the data will be accessed.

    Proxies use something similar to `functools.partial` to create a callable class instead of returning the raw data (see https://github.com/brightway-lca/bw_processing/issues/9 for why we can't just use `partial`). datapackage access methods (i.e. `.get_resource`) will automatically resolve proxies when needed.

    Args:

        * fs_or_obj. A `Filesystem` or an instance of `DatapackageBase`.
        * mmap_mode. `str`, optional. Define memory mapping mode to use when loading Numpy arrays.
        * proxy. bool, default `False`. Load proxies instead of complete Numpy arrays; see above.

    Returns:

        A `Datapackage` instance.

    """
    if isinstance(fs_or_obj, DatapackageBase):
        obj = fs_or_obj
    else:
        obj = Datapackage()
        obj._load(fs=fs_or_obj, mmap_mode=mmap_mode, proxy=proxy)
    return obj
