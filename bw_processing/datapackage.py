from .constants import DEFAULT_LICENSES
from .errors import Closed, LengthMismatch, NonUnique, InvalidMimetype
from .filesystem import clean_datapackage_name
from .io_helpers import file_writer, file_reader
from .proxies import UndefinedInterface
from .utils import (
    check_name,
    check_suffix,
    load_bytes,
    resolve_dict_iterator,
    NoneSorter,
)
from copy import deepcopy
from fs.base import FS
from fs.memoryfs import MemoryFS
from functools import partial
from pathlib import Path
from typing import Union, Any
import datetime
import numpy as np
import pandas as pd
import uuid


class DatapackageBase:
    def __init__(self):
        self._finalized = False
        self._cache = {}

    def __get_resources(self) -> list:
        return self.metadata["resources"]

    def __set_resources(self, dct: dict) -> None:
        self.metadata["resources"] = dct

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
        index = self._get_index(name_or_index)

        try:
            self.fs.remove(self.resources["path"])
        except KeyError:
            # Interface has no path
            pass

        del self.metadata["resources"][index]
        del self.data[index]

    def get_resource(self, name_or_index: Union[str, int]) -> (Any, dict):
        """Return data and metadata for ``name_or_index``.

        Args:
            name_or_index: Name (str) or index (int) of a resource in the existing metadata.

        Raises:
            IndexError: Integer index out of range of given metadata
            ValueError: String name not present in metadata
            NonUnique: String name present in two resource metadata sections

        Returns:
            Metadata dict
        """
        try:
            self._cache[name_or_index]
        except KeyError:
            index = self._get_index(name_or_index)

            if isinstance(self.data[index], partial):
                obj = self.data[index]()
                self.data[index] = obj

            self._cache[name_or_index] = (self.data[index], self.resources[index])
        return self._cache[name_or_index]

    def filter_by_attribute(self, key: str, value: Any) -> "FilteredDatapackage":
        """"""
        fdp = FilteredDatapackage()
        fdp.metadata = deepcopy(self.metadata)
        intermediate = list(
            zip(
                *[
                    (array, resource)
                    for array, resource in zip(self.data, self.resources)
                    if resource.get(key) == value
                ]
            )
        )
        fdp.data, fdp.resources = list(intermediate[0]), list(intermediate[1])
        return fdp


class FilteredDatapackage(DatapackageBase):
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

    # def del_package(self):
    #     """Delete this data package, including any saved data. Frees up any memory used by data resources."""
    #     self.io_obj.delete_all()
    #     self.io_obj = self.resources = self.data = None

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
        sum_duplicates: bool = False,
        substitute: bool = True,
    ) -> None:
        """Start a new data package.

        All metadata elements should follow the `datapackage specification <https://frictionlessdata.io/specs/data-package/>`__.

        Licenses are specified as a list in ``metadata``. The default license is the `Open Data Commons Public Domain Dedication and License v1.0 <http://opendatacommons.org/licenses/pddl/>`__.
        """
        name = clean_datapackage_name(name or uuid.uuid4().hex)
        check_name(name)

        self.fs = fs or MemoryFS()

        # if dirpath is None:
        #     self.io_obj = InMemoryIO()
        # elif compress:
        #     self.io_obj = TemporaryDirectoryIO(
        #         dest_dirpath=dirpath,
        #         dest_filename=check_suffix(name, ".zip"),
        #         overwrite=overwrite,
        #     )
        # else:
        #     self.io_obj = DirectoryIO(dirpath=dirpath, overwrite=overwrite)

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
            "sum_duplicates": sum_duplicates,
            "substitute": substitute,
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
        if name in existing_names:
            raise NonUnique("This name already used")

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
        """"""
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
        """"""
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

    # def add_structured_array(
    #     self,
    #     iterable_data_source: Any,
    #     matrix: str,
    #     name: Union[str, None] = None,
    #     nrows: Union[int, None] = None,
    #     dtype: Any = None,
    #     extra: Union[None, dict] = None,
    #     is_interface: bool = False,
    # ) -> None:
    #     """Add a numpy structured array resource that will be used to create at least part of a matrix.

    #     ``iterable_data_source`` can be any of the following:

    #     * An iterable (i.e. an object that supports the python iterator interface) that will return rows that can be used to create a numpy array. This is the most common use case, with the iterable being a wrapped database cursor. This object must return rows as tuples which match the dtype of the structured array. I strongly encourage using the common datatype (``COMMON_DTYPE``), which is also the default.
    #     * A numpy structured array.
    #     * A numpy structured array serialized into a ``BytesIO`` object.

    #     In contrast with presamples arrays, ``iterable_data_source`` cannot be an infinite generator. We need a finite set of data to build a matrix.

    #     A file will be created on disk if both of the following hold:

    #         * The ``Datapackage.io_obj`` object is not ``InMemoryIO``, i.e. the call to ``Datapackage.create`` had a real file or directory path.
    #         * ``is_interface`` is false.

    #     Memory notes:

    #         * This method will not retain a reference to ``iterable_data_source``.
    #         * If the array is written to disk, a ``functools.partial`` object is added to ``self.data``. Once a ``functools.partial`` object is called, it is loaded into memory (and the object in ``self.data`` is substituted by the loaded array).
    #         * If the data package is created in memory, the entire array is kept in memory.

    #     Args:

    #         * iterable_data_source: See discussion above
    #         * matrix: The label of the matrix to be constructed
    #         * name (optional): The name of this resource. Names must be unique in a given data package
    #         * nrows (optional): Number of rows in array. You gain a bit of speed and memory if this is specified ahead of time.
    #         * dtype (optional, default is COMMON_DTYPE): Numpy dtype of created array
    #         * extra (optional): Dict of extra metadata
    #         * is_interface (optional): Flag indicating whether this resource is an interface to an external data source. Interfaces are never saved to disk.

    #     Returns:

    #         Nothing, but appends objects to ``self.metadata['resources']`` and ``self.data``.

    #     """
    #     data, name = self._prepare_modifications(iterable_data_source, name)

    #     filename = check_suffix(name, ".npy")

    #     if isinstance(iterable_data_source, np.ndarray):
    #         array = iterable_data_source
    #     else:
    #         array = create_structured_array(iterable_data_source, nrows, dtype)

    #     if not is_interface:
    #         self.io_obj.save_numpy(array, filename)
    #         self.data.append(self.io_obj.load_numpy(filename, proxy=True))
    #     else:
    #         self.data.append(array)

    #     resource = {
    #         # Datapackage generic
    #         "profile": "interface" if is_interface else "data-resource",
    #         "format": "npy",
    #         "mediatype": "application/octet-stream",
    #         "name": name,
    #         # Brightway specific
    #         "matrix": matrix,
    #         "kind": "processed array",
    #     }
    #     if not is_interface:
    #         resource["path"] = str(filename)
    #     self._add_extra_metadata(resource, extra)
    #     self.resources.append(resource)

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

    # def add_presamples_indices_array(
    #     self,
    #     iterable_data_source: Any,
    #     data_array: str,
    #     name: Union[str, None] = None,
    #     nrows: Union[int, None] = None,
    #     extra: Union[None, dict] = None,
    # ) -> None:
    #     """Add a numpy structured array resource that will be used to identify row and column matrix indices of presamples data.

    #     Each presamples data array which will be used in modifying matrices must have a corresponding indices array.

    #     ``iterable_data_source`` can be any of the following:

    #     * An iterable (i.e. an object that supports the python iterator interface) that will return rows that can be used to create a numpy array. This is the most common use case, with the iterable being a wrapped database cursor. This object must return rows as tuples which match the indices dtype (``constants.INDICES_DTYPE``).
    #     * A numpy structured array which has the dtype ``constants.INDICES_DTYPE``.
    #     * A numpy structured array serialized into a ``BytesIO`` object which has the dtype ``constants.INDICES_DTYPE``.

    #     Args:

    #         * iterable_data_source: See discussion above matrix: The label of the matrix to be constructed
    #         * name (optional): The name of this resource. Names must be unique in a given data package
    #         * nrows (optional): Number of rows in array. You gain a bit of speed and memory if this is specified ahead of time.
    #         * dtype (optional, default is COMMON_DTYPE): Numpy dtype of created array
    #         * extra (optional): Dict of extra metadata
    #         * is_interface (optional): Flag indicating whether this resource is an interface to an external data source. Interfaces are never saved to disk.

    #     Returns:

    #         Nothing, but appends objects to ``self.metadata['resources']`` and ``self.data``.

    #     """
    #     assert isinstance(data_array, str)
    #     assert data_array in {obj["name"] for obj in self.resources}

    #     data, name = self._prepare_modifications(iterable_data_source, name)

    #     filename = check_suffix(name, ".npy")

    #     if isinstance(iterable_data_source, np.ndarray):
    #         array = iterable_data_source
    #     else:
    #         array = create_structured_indices_array(iterable_data_source, nrows)

    #     self.io_obj.save_numpy(array, filename)
    #     self.data.append(self.io_obj.load_numpy(filename, proxy=True))

    #     resource = {
    #         # Datapackage generic
    #         "profile": "data-resource",
    #         "format": "npy",
    #         "mediatype": "application/octet-stream",
    #         "name": name,
    #         "path": str(filename),
    #         # Brightway specific
    #         "data_array": data_array,
    #     }
    #     self._add_extra_metadata(resource, extra)
    #     self.resources.append(resource)

    # def add_presamples_data_array(
    #     self,
    #     iterable_data_source: Any,
    #     matrix: str,
    #     name: Union[None, str] = None,
    #     nrows: Union[None, int] = None,
    #     dtype: Any = np.float32,
    #     extra: Union[None, dict] = None,
    #     is_interface: bool = False,
    # ) -> None:
    #     data, name = self._prepare_modifications(iterable_data_source, name)

    #     filename = check_suffix(name, ".npy")

    #     if is_interface:
    #         self.data.append(iterable_data_source)
    #     else:
    #         array = create_array(iterable_data_source, nrows, dtype)

    #         self.io_obj.save_numpy(array, filename)
    #         self.data.append(self.io_obj.load_numpy(filename, proxy=True))

    #     resource = {
    #         # Datapackage generic
    #         "profile": "interface" if is_interface else "data-resource",
    #         "format": "npy",
    #         "mediatype": "application/octet-stream",
    #         "name": name,
    #         # Brightway specific
    #         "matrix": matrix,
    #         "kind": "presamples",
    #     }
    #     if not is_interface:
    #         resource["path"] = str(filename)
    #     self._add_extra_metadata(resource, extra)
    #     self.resources.append(resource)


def create_datapackage(
    fs: Union[FS, None] = None,
    name: Union[None, str] = None,
    id_: Union[None, str] = None,
    metadata: Union[dict, None] = None,
    combinatorial: bool = False,
    sequential: bool = False,
    seed: Union[int, None] = None,
    sum_duplicates: bool = True,
    substitute: bool = True,
) -> Datapackage:
    obj = Datapackage()
    obj._create(
        fs=fs,
        name=name,
        id_=id_,
        metadata=metadata,
        sequential=sequential,
        combinatorial=combinatorial,
        seed=seed,
        sum_duplicates=sum_duplicates,
        substitute=substitute,
    )
    return obj


def load_datapackage(
    fs_or_obj: Union[DatapackageBase, Path, str],
    mmap_mode: Union[None, str] = None,
    proxy: bool = False,
) -> Datapackage:
    if isinstance(fs_or_obj, DatapackageBase):
        obj = fs_or_obj
    else:
        obj = Datapackage()
        obj._load(fs=fs_or_obj, mmap_mode=mmap_mode, proxy=proxy)
    return obj
