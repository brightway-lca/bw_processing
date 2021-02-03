from bw_processing.filesystem import safe_filename, md5, clean_datapackage_name
from pathlib import Path
import platform
import pytest

fixtures_dir = Path(__file__, "..").resolve() / "fixtures"
_windows = platform.system() == "Windows"


@pytest.mark.skipif(_windows, reason="Different line encodings on Windows")
def test_md5_text():
    assert md5(fixtures_dir / "lorem.txt") == "edc715389af2498a623134608ba0a55b"


def test_md5_binary():
    assert md5(fixtures_dir / "array.npy") == "bbadddf09cf6b1e36d8333f474e36cee"


def test_safe_filename():
    assert safe_filename("Wave your hand yeah 🙋!") == "Wave-your-hand-yeah.f7952a3d"
    assert (
        safe_filename("Wave your hand yeah 🙋!", full=True)
        == "Wave-your-hand-yeah.f7952a3d4b0534cdac0e0cbbf66aac73"
    )
    assert (
        safe_filename("Wave your hand yeah 🙋!", add_hash=False) == "Wave-your-hand-yeah"
    )


def test_clean_datapackage_name():
    a = "('IPCC', 'simple') processed arrays"
    b = "IPCC_simple_processed_arrays"
    assert clean_datapackage_name(a) == b
