from .array_creation import (
    create_array,
    create_structured_array,
    create_structured_indices_array,
)
from .constants import DEFAULT_LICENSES
from .errors import Closed, LengthMismatch, NonUnique
from .filesystem import clean_datapackage_name
from .io_classes import InMemoryIO, ZipfileIO, TemporaryDirectoryIO, DirectoryIO
from .proxies import ReadProxy, GenericProxy
from .utils import check_name, check_suffix, load_bytes
from pathlib import Path
from typing import Union, Any
import datetime
import numpy as np
import pandas as pd
import uuid


class Datapackage:
    """
    Interface for creating, loading, and using numerical datapackages for Brightway.

    Note that there are two entry points to using this class: ``Datapackage.create()`` and ``Datapackage.load()``. Do not create an instance of the class with ``Datapackage()``, it almost certainly won't work!

    Data packages can be stored in memory, in a directory, or in a zip file. When creating data packages for use later, don't forget to call ``.finalize()``, or the metadata won't be written and the data package won't be usable.

    Potential gotchas:

    * There is currently no way to modify a zipped data package once it is finalized.
    * Resources that are interfaces to external data sources (either in Python or other) can't be saved, but must be recreated each time a data package is used.

    """

    def __init__(self):
        self._finalized = False

    def _check_length_consistency(self) -> None:
        if len(self.resources) != len(self.data):
            raise LengthMismatch(
                "Number of resources ({}) doesn't match number of data objects ({})".format(
                    len(self.resources), len(self.data)
                )
            )

    def __get_resources(self):
        return self.metadata["resources"]

    def __set_resources(self, dct: dict):
        self.metadata["resources"] = dct

    resources = property(__get_resources, __set_resources)

    @staticmethod
    def load(path: Union[Path, str], mmap_mode: Union[None, str] = None) -> Datapackage:
        obj = Datapackage()
        obj._load(path, mmap_mode)
        return obj

    def _load(self, path: Union[Path, str], mmap_mode: Union[None, str]) -> None:
        path = Path(path)
        if not path.exists():
            raise ValueError("Given path doesn't exist")
        if path.is_file():
            self.io_obj = ZipfileIO(path)
        elif path.is_dir():
            self.io_obj = DirectoryIO(path, new=False)
        else:
            raise ValueError("Can't understand given path")

        self.metadata = self.io_obj.load_json("datapackage.json")
        self.data = []
        self._load_all(mmap_mode)

    def _load_all(self, mmap_mode: Union[None, str]) -> None:
        for resource in self.resources:
            if (
                resource["mediatype"] == "application/octet-stream"
                and resource["format"] == "npy"
            ):
                self.data.append(
                    self.io_obj.load_numpy(resource["path"], mmap_mode=mmap_mode)
                )
            elif resource["mediatype"] == "text/csv":
                self.data.append(
                    self.io_obj.load_csv(resource["path"], mmap_mode=mmap_mode)
                )
            elif resource["mediatype"] == "application/json":
                self.data.append(
                    self.io_obj.load_json(resource["path"], mmap_mode=mmap_mode)
                )
            else:
                self.data.append(GenericProxy())

    @staticmethod
    def create(
        dirpath: Union[Path, str] = None,
        name: Union[None, str] = None,
        id_: Union[None, str] = None,
        metadata: Union[dict, str] = None,
        overwrite: bool = False,
        compress: bool = False,
    ) -> Datapackage:
        obj = Datapackage()
        obj._create(dirpath, name, id_, metadata, compress)
        return obj

    def _create(
        self,
        dirpath: Union[str, Path, None],
        name: Union[str, None],
        id_: Union[str, None],
        metadata: Union[dict, None],
        overwrite: bool = False,
        compress: bool,
    ) -> None:
        """Start a new data package.

        All metadata elements should follow the `datapackage specification <https://frictionlessdata.io/specs/data-package/>`__.

        Licenses are specified as a list in ``metadata``. The default license is the `Open Data Commons Public Domain Dedication and License v1.0 <http://opendatacommons.org/licenses/pddl/>`__.
        """
        name = clean_datapackage_name(name or uuid.uuid4().hex)
        check_name(name)

        if dirpath is None:
            self.io_obj = InMemoryIO()
        elif compress:
            self.io_obj = TemporaryDirectoryIO(dirpath, check_suffix(name, ".zip"))
        else:
            self.io_obj = DirectoryIO(dirpath)

        self.metadata = {
            "profile": "data-package",
            "name": name,
            "id": id_ or uuid.uuid4().hex,
            "licenses": (metadata or {}).get("licenses", DEFAULT_LICENSES),
            "resources": [],
            "created": datetime.datetime.utcnow().isoformat("T") + "Z",
        }
        for k, v in (metadata or {}).items():
            if k not in self.metadata:
                self.metadata[k] = v

        self.data = []

    def _purge_interfaces(self) -> None:
        """Remove resources with the profile ``interface``, as they can't be saved to disk."""
        interface_indices = [
            index
            for index, obj in enumerate(self.resources)
            if obj["profile"] == "interface"
        ]
        self.resources = [
            obj
            for index, obj in enumerate(self.resources)
            if index not in interface_indices
        ]
        self.data = [
            obj for index, obj in enumerate(self.data) if index not in interface_indices
        ]

    def finalize(self) -> None:
        self._purge_interfaces()
        self._check_length_consistency()

        if self._finalized:
            raise Closed("Datapackage already finalized")

        self.io_obj.save_json(self.metadata, "datapackage.json")
        self.io_obj.archive()

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
        if isinstance(name_or_index, int):
            if name_or_index >= len(self.resources):
                raise IndexError(
                    "Index {} given, but only {} resources available".format(
                        name_or_index, len(self.resources)
                    )
                )
            index = name_or_index
        else:
            indices = []
            for i, o in enumerate(self.metadata):
                if o["name"] == name_or_index:
                    indices.append(i)

            if not indices:
                raise ValueError("Name {} not found in metadata".format(name_or_index))
            elif len(indices) > 1:
                raise NonUnique("This name present at indices: {}".format(indices))
            else:
                index = indices[0]

        if isinstance(self.data[index], ReadProxy):
            obj = self.data[index]()
            self.data[index] = obj

        return self.data[index], self.resources[index]

    def _prepare_modifications(self, data: Any, name: str) -> (str, Any):
        data = load_bytes(data)

        if self._finalized:
            raise Closed("Datapackage already finalized")

        self._check_length_consistency()

        name = name or uuid.uuid4().hex
        check_name(name)

        existing_names = {o["name"] for o in self.resources}
        if name in existing_names:
            raise NonUnique("This name already used")

        return name, data

    def _add_extra_metadata(self, resource: dict, extra: Union[dict, None]) -> None:
        for key in extra or {}:
            if key not in resource:
                resource[key] = extra[key]

    def add_structured_array(
        self,
        iterable_data_source: Any,
        matrix_label: str,
        name: Union[str, None] = None,
        nrows: Union[int, None] = None,
        dtype: Any = None,
        extra: Union[None, dict] = None,
        is_interface: bool = False,
    ) -> None:
        """Add a numpy structured array resource that will be used to create at least part of a matrix.

        ``iterable_data_source`` can be any of the following:

        * An iterable (i.e. an object that supports the python iterator interface) that will return rows that can be used to create a numpy array. This is the most common use case, with the iterable being a wrapped database cursor. This object must return rows as tuples which match the dtype of the structured array. I strongly encourage using the common datatype (``COMMON_DTYPE``), which is also the default.
        * A numpy structured array.
        * A numpy structured array serialized into a ``BytesIO`` object.

        In contrast with presamples arrays, ``iterable_data_source`` cannot be an infinite generator. We need a finite set of data to build a matrix.

        A file will be created on disk if both of the following hold:

            * The ``Datapackage.io_obj`` object is not ``InMemoryIO``, i.e. the call to ``Datapackage.create`` had a real file or directory path.
            * ``is_interface`` is false.

        Memory notes:

            * This method will not retain a reference to ``iterable_data_source``.
            * If the array is written to disk, a ``ReadProxy`` object is added to ``self.data``. Once a ``ReadProxy`` object is accessed using ``self.get_resource()``, it is loaded into memory (and the object in ``self.data`` is substituted by the loaded array).
            * If the data package is created in memory, the entire array is kept in memory.

        Args:

            * iterable_data_source: See discussion above
            * matrix_label: The label of the matrix to be constructed
            * name (optional): The name of this resource. Names must be unique in a given data package
            * nrows (optional): Number of rows in array. You gain a bit of speed and memory if this is specified ahead of time.
            * dtype (optional, default is COMMON_DTYPE): Numpy dtype of created array
            * extra (optional): Dict of extra metadata
            * is_interface (optional): Flag indicating whether this resource is an interface to an external data source. Interfaces are never saved to disk.

        Returns:

            Nothing, but appends objects to ``self.metadata['resources']`` and ``self.data``.

        """
        name, data = self._prepare_modifications(iterable_data_source, name)

        filename = check_suffix(name, ".npy")

        if isinstance(iterable_data_source, np.ndarray):
            array = iterable_data_source
        else:
            array = create_structured_array(iterable_data_source, nrows, dtype)

        if not is_interface:
            self.io_obj.save_numpy(array, filename)
            self.data.append(self.io_obj.load_numpy(filename, proxy=True))
        else:
            self.data.append(array)

        resource = {
            # Datapackage generic
            "profile": "interface" if is_interface else "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": name,
            # Brightway specific
            "matrix": matrix_label,
            "kind": "processed array",
        }
        if not is_interface:
            resource["path"] = str(filename)
        self._add_extra_metadata(resource, extra)
        self.resources.append(resource)

    def add_csv_metadata(
        self,
        dataframe: pd.DataFrame,
        valid_for: list,
        name: str = None,
        extra: dict = None,
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

        names = {obj["name"] for obj in self.resources}
        assert all(x in names for x, y in valid_for)

        name, data = self._prepare_modifications(dataframe, name)

        filename = check_suffix(name, ".csv")

        self.io_obj.save_csv(dataframe, filename)
        self.data.append(dataframe)

        resource = {
            # Datapackage generic
            "profile": "data-resource",
            "mediatype": "text/csv",
            "path": str(filename),
            "name": name,
            # Brightway specific
            "valid_for": valid_for,
        }
        for key in extra or {}:
            if key not in resource:
                resource[key] = extra[key]

        self.resources.append(resource)

    def add_json_metadata(
        self, data: Any, data_array: str, name: str = None, extra: dict = None
    ):
        """Add an iterable metadata object to be stored as a JSON file.

        The purpose of storing metadata is to enable data exchange; therefore, this method assumes that data is written to disk.

        The normal use case of this method is to provide names and other metadata for parameters whose values are stored as presamples arrays. The length of ``data`` should match the number of rows in the corresponding presamples array, and ``data`` is just a list of string labels for the parameters.

        In contrast to ``self.create_structured_array``, this always stores the dataframe in ``self.data``; no proxies are used.

        Args:

            * data: Data to be persisted to disk.
            * data_array: Name of presample array that this metadata is valid for.
            * name (optional): The name of this resource. Names must be unique in a given data package
            * extra (optional): Dict of extra metadata

        Returns:

            Nothing, but appends objects to ``self.metadata['resources']`` and ``self.data``.

        Raises:

            * AssertionError: If inputs are not in correct form
            * AssertionError: If ``data_array`` refers to unavailable resources

        """
        assert isinstance(data_array, str)
        assert data_array in {obj["name"] for obj in self.resources}

        data = self._prepare_modifications(data)

        name = name or uuid.uuid4().hex
        check_name(name)

        filename = check_suffix(name, ".json")

        self.io_obj.save_json(data, filename)
        self.data.append(data)

        resource = {
            # Datapackage generic
            "profile": "data-resource",
            "mediatype": "application/json",
            "path": str(filename),
            "name": name,
            # Brightway specific
            "data_array": data_array,
        }
        for key in extra or {}:
            if key not in resource:
                resource[key] = extra[key]

        self.resources.append(resource)

    def add_presamples_indices_array(
        self,
        iterable_data_source: Any,
        data_array: str,
        name: Union[str, None] = None,
        nrows: Union[int, None] = None,
        extra: Union[None, dict] = None,
    ) -> None:
        """Add a numpy structured array resource that will be used to identify row and column matrix indices of presamples data.

        Each presamples data array which will be used in modifying matrices must have a corresponding indices array.

        ``iterable_data_source`` can be any of the following:

        * An iterable (i.e. an object that supports the python iterator interface) that will return rows that can be used to create a numpy array. This is the most common use case, with the iterable being a wrapped database cursor. This object must return rows as tuples which match the indices dtype (``constants.INDICES_DTYPE``).
        * A numpy structured array which has the dtype ``constants.INDICES_DTYPE``.
        * A numpy structured array serialized into a ``BytesIO`` object which has the dtype ``constants.INDICES_DTYPE``.

        I strongly recommend using the utility function ``indices_wrapper``.

        Args:

            * iterable_data_source: See discussion above matrix_label: The label of the matrix to be constructed
            * name (optional): The name of this resource. Names must be unique in a given data package
            * nrows (optional): Number of rows in array. You gain a bit of speed and memory if this is specified ahead of time.
            * dtype (optional, default is COMMON_DTYPE): Numpy dtype of created array
            * extra (optional): Dict of extra metadata
            * is_interface (optional): Flag indicating whether this resource is an interface to an external data source. Interfaces are never saved to disk.

        Returns:

            Nothing, but appends objects to ``self.metadata['resources']`` and ``self.data``.

        """
        assert isinstance(data_array, str)
        assert data_array in {obj["name"] for obj in self.resources}

        name, data = self._prepare_modifications(iterable_data_source, name)

        filename = check_suffix(name, ".npy")

        if isinstance(iterable_data_source, np.ndarray):
            array = iterable_data_source
        else:
            array = create_structured_indices_array(iterable_data_source, nrows)

        self.io_obj.save_numpy(array, filename)
        self.data.append(self.io_obj.load_numpy(filename, proxy=True))

        resource = {
            # Datapackage generic
            "profile": "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": name,
            "path": str(filename),
            # Brightway specific
            "data_array": data_array,
        }
        self._add_extra_metadata(resource, extra)
        self.resources.append(resource)

    def add_presamples_data_array(
        self,
        iterable_data_source: Any,
        matrix_label: str,
        name: Union[None, str] = None,
        nrows: Union[None, int] = None,
        dtype: Any = np.float32,
        extra: Union[None, dict] = None,
        is_interface: bool = False,
    ) -> None:
        name, data = self._prepare_modifications(iterable_data_source, name)

        filename = check_suffix(name, ".npy")

        if is_interface:
            self.data.append(iterable_data_source)
        else:
            array = create_array(iterable_data_source, nrows, dtype)

            self.io_obj.save_numpy(array, filename)
            self.data.append(self.io_obj.load_numpy(filename, proxy=True))

        resource = {
            # Datapackage generic
            "profile": "interface" if is_interface else "data-resource",
            "format": "npy",
            "mediatype": "application/octet-stream",
            "name": name,
            # Brightway specific
            "matrix": matrix_label,
            "kind": "presamples",
        }
        if not is_interface:
            resource["path"] = str(filename)
        self._add_extra_metadata(resource, extra)
        self.resources.append(resource)
