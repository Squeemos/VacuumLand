import pytest
from .vacuum_land import VacuumLand

def test_width():
    with pytest.raises(AssertionError):
        v = VacuumLand(width = False)

def test_width_float():
    with pytest.raises(AssertionError):
        v = VacuumLand(width = 10.1)

def test_width_list():
    with pytest.raises(AssertionError):
        v = VacuumLand(width = [])

def test_width_zero():
    with pytest.raises(AssertionError):
        v = VacuumLand(width = 0)
