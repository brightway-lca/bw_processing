from bw_processing.proxies import Proxy
from pathlib import Path
import numpy as np
import io


dirpath = Path(__file__).parent.resolve() / "fixtures"


def test_proxy_rewinds_file_object():
    p = Proxy(np.load, "file", {"file": open(dirpath / "array.npy", "rb")})
    first = p()
    second = p()
    assert np.allclose(first, second)


def test_proxy_rewinds_buffer():
    arr = np.random.random(size=(10, 10))
    stream = io.BytesIO()
    np.save(stream, arr, allow_pickle=False)

    p = Proxy(np.load, "file", {"file": stream})
    first = p()
    second = p()
    assert np.allclose(first, arr)
    assert np.allclose(first, second)
