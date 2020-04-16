from bw_processing.filesystem import safe_filename, md5
from pathlib import Path

fixtures_dir = Path(__file__, "..").resolve() / "fixtures"


def test_md5():
    assert md5(fixtures_dir / "lorem.txt") == "db89bb5ceab87f9c0fcc2ab36c189c2c"


def test_safe_filename():
    assert (
        safe_filename("Wave your hand yeah 🙋!")
        == "Wave-your-hand-yeah.f7952a3d4b0534cdac0e0cbbf66aac73"
    )
    assert (
        safe_filename("Wave your hand yeah 🙋!", add_hash=False) == "Wave-your-hand-yeah"
    )
