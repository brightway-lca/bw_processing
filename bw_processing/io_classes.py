from .proxies import ReadProxy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tempfile
import zipfile


class IOBase:
    def __init__(self):
        pass

    def archive(self):
        pass


class DirectoryIO(IOBase):
    def __init__(self, dirpath):
        self.dirpath = Path(dirpath)
        if not self.dirpath.parent.is_dir():
            raise ValueError(
                "Parent directory `{}` doesn't exist".format(self.dirpath.parent)
            )
        if self.dirpath.is_dir():
            raise ValueError("Directory `{}` already exists".format(self.dirpath))
        self.dirpath.mkdir()

    def load_json(self, filename, proxy=False, mmap_mode=None):
        return json.load(open(self.dirpath / filename))

    def save_json(self, data, filename):
        with open(self.dirpath / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_numpy(self, filename, proxy=False, mmap_mode=None):
        kwargs = {
            "file": self.dirpath / filename,
            "mmap_mode": mmap_mode,
            "allow_pickle": False,
        }
        if not proxy:
            return np.load(**kwargs)
        else:
            return ReadProxy(np.load, **kwargs)

    def save_numpy(self, array, filename):
        np.save(self.dirpath / filename, array, allow_pickle=False)

    def load_csv(self, filename, proxy=False, mmap_mode=None):
        return pd.read_csv(open(self.dirpath / filename))

    def save_csv(self, data, filename):
        assert isinstance(data, pd.DataFrame)
        data.to_csv(self.dirpath / filename)


class ZipfileIO(IOBase):
    def __init__(self, filepath):
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path)

    def load_numpy(self, filename, proxy=False, mmap_mode=None):
        kwargs = {
            "file": self.zf.open(filename),
            "mmap_mode": mmap_mode,
            "allow_pickle": False,
        }
        if not proxy:
            return np.load(**kwargs)
        else:
            return ReadProxy(np.load, **kwargs)

    def save_numpy(self, *args):
        raise NotImplemented("Read-only zipfile")

    def load_json(self, filename, proxy=False, mmap_mode=None):
        return json.load(open(self.dirpath / filename))

    def save_json(self, *args):
        raise NotImplemented("Read-only zipfile")

    def load_csv(self, filename, proxy=False, mmap_mode=None):
        return pd.read_csv(self.zf.open(filename))

    def save_csv(self, *args, **kwargs):
        raise NotImplemented("Read-only zipfile")

    def archive(self):
        raise NotImplemented("Read-only zipfile")


class TemporaryDirectoryIO(DirectoryIO):
    def __init__(self, dest_dirpath, dest_filename):
        self.dest_dirpath = Path(dest_dirpath)
        if not self.dest_dirpath.is_dir():
            raise ValueError(
                "Destination directory `{}` doesn't exist".format(self.dest_dirpath)
            )
        self.dest_filepath = self.dest_dirpath / dest_filename
        if self.dest_filepath.is_file():
            raise ValueError("This calculation package archive already exists")

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
