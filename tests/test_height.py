import pytest
from .vacuum_land import VacuumLand

def test_height_bool():
    with pytest.raises(AssertionError):
        v = VacuumLand(height = False)

def test_height_float():
    with pytest.raises(AssertionError):
        v = VacuumLand(height = 10.1)

def test_height_list():
    with pytest.raises(AssertionError):
        v = VacuumLand(height = [])

def test_height_zero():
    with pytest.raises(AssertionError):
        v = VacuumLand(height = 0)
