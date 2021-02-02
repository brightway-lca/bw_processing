from bw_processing.filesystem import safe_filename, md5, clean_datapackage_name
from pathlib import Path

fixtures_dir = Path(__file__, "..").resolve() / "fixtures"


def test_md5():
    assert md5(fixtures_dir / "test-fixture.zip") == "5f649cabcbc98da53f4176f1968eb662"


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
