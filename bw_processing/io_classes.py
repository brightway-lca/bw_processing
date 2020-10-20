from pathlib import Path
import json
import numpy as np
import pandas as pd
import tempfile
import zipfile
from functools import partial


class IOBase:
    def __init__(self):
        pass

    def archive(self):
        pass

    def _load_resource(self, func, kwargs, proxy=False):
        if proxy:
            return func(**kwargs)
        else:
            return partial(func, **kwargs)


class DirectoryIO(IOBase):
    def __init__(self, dirpath, overwrite=False):
        self.dirpath = Path(dirpath)
        if self.dirpath.is_dir():
            if not overwrite:
                raise ValueError(
                    "Directory `{}` already exists and `overwrite` is false.".format(
                        self.dirpath
                    )
                )
        else:
            if not self.dirpath.parent.is_dir():
                raise ValueError(
                    "Parent directory `{}` doesn't exist".format(self.dirpath.parent)
                )
            self.dirpath.mkdir()

    def delete_file(self, filename):
        (self.dirpath / filename).unlink()

    def load_json(self, filename, proxy=False, mmap_mode=None):
        return self._load_resource(
            json.load, {"fp": open(self.dirpath / filename)}, proxy
        )

    def save_json(self, data, filename):
        with open(self.dirpath / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_numpy(self, filename, proxy=False, mmap_mode=None):
        kwargs = {
            "file": self.dirpath / filename,
            "mmap_mode": mmap_mode,
            "allow_pickle": False,
        }
        return self._load_resource(np.load, kwargs, proxy)

    def save_numpy(self, array, filename):
        np.save(self.dirpath / filename, array, allow_pickle=False)

    def load_csv(self, filename, proxy=False, mmap_mode=None):
        return self._load_resource(
            pd.read_csv, {"filepath_or_buffer": open(self.dirpath / filename)}, proxy
        )

    def save_csv(self, data, filename):
        assert isinstance(data, pd.DataFrame)
        data.to_csv(self.dirpath / filename)


class ZipfileIO(IOBase):
    def __init__(self, filepath):
        self.path = Path(filepath)
        self.zf = zipfile.ZipFile(self.path)

    def delete_file(self, filename):
        raise NotImplementedError("Read-only zipfile")

    def load_numpy(self, filename, proxy=False, mmap_mode=None):
        kwargs = {
            "file": self.zf.open(filename),
            "mmap_mode": mmap_mode,
            "allow_pickle": False,
        }
        return self._load_resource(np.load, kwargs, proxy)

    def save_numpy(self, *args):
        raise NotImplementedError("Read-only zipfile")

    def load_json(self, filename, proxy=False, mmap_mode=None):
        return self._load_resource(json.load, {"fp": self.zf.open(filename)}, proxy)

    def save_json(self, *args):
        raise NotImplementedError("Read-only zipfile")

    def load_csv(self, filename, proxy=False, mmap_mode=None):
        return self._load_resource(
            pd.read_csv, {"filepath_or_buffer": self.zf.open(filename)}, proxy
        )

    def save_csv(self, *args, **kwargs):
        raise NotImplementedError("Read-only zipfile")

    def archive(self):
        raise NotImplementedError("Read-only zipfile")


class TemporaryDirectoryIO(DirectoryIO):
    def __init__(self, dest_dirpath, dest_filename, overwrite=False):
        self.dest_dirpath = Path(dest_dirpath)
        if not self.dest_dirpath.is_dir():
            raise ValueError(
                "Destination directory `{}` doesn't exist".format(self.dest_dirpath)
            )
        self.dest_filepath = self.dest_dirpath / dest_filename
        if self.dest_filepath.is_file():
            if overwrite:
                self.dest_filepath.unlink()
            else:
                raise ValueError(
                    "This calculation package archive already exists and `overwrite` is false."
                )

        self.td = tempfile.TemporaryDirectory()
        self.dirpath = Path(self.td.name)

    def archive(self):
        with zipfile.ZipFile(
            self.dest_filepath, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for file in self.dirpath.iterdir():
                if file.is_file():
                    zf.write(file, arcname=file.name)
        del self.td


class InMemoryIO(IOBase):
    def __init__(self):
        self._np_cache = {}
        self._json_cache = {}
        self._csv_cache = {}

    def delete_file(self, filename):
        for cache in (self._np_cache, self._json_cache, self._csv_cache):
            if filename in cache:
                del cache[filename]
                break

    def load_json(self, filename, *args, **kwargss):
        return self._json_cache[filename]

    def save_json(self, data, filename):
        self._json_cache[filename] = data

    def load_numpy(self, filename, *args, **kwargss):
        return self._np_cache[filename]

    def save_numpy(self, array, filename):
        self._np_cache[filename] = array

    def load_csv(self, filename, *args, **kwargss):
        return self._csv_cache[filename]

    def save_csv(self, data, filename):
        self._csv_cache[filename] = data
