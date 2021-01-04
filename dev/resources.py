from abc import ABC, abstractmethod
import json
import numpy


class WriteableResource(ABC):
    @abstractmethod
    def write(self, path):
        raise NotImplemented


class JSONResource(WriteableResource):
    def __init__(self, data):
        self.data = data

    def write(self, path):
        if path is None:
            return self.data
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)


class NumpyResource(WriteableResource):
    def __init__(self, data):
        self.data = data

    def write(self, path):
        if path is None:
            return self.data
        else:
            numpy.save(path, self.data, allow_pickle=False)
