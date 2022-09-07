import pytest
from .vacuum_land import VacuumLand

def test_height():
    with pytest.raises(AssertionError):
        v = VacuumLand(height = False)
    with pytest.raises(AssertionError):
        v = VacuumLand(height = 10.1)
    with pytest.raises(AssertionError):
        v = VacuumLand(height = [])

def test_width():
    with pytest.raises(AssertionError):
        v = VacuumLand(width = False)
    with pytest.raises(AssertionError):
        v = VacuumLand(width = 10.1)
    with pytest.raises(AssertionError):
        v = VacuumLand(width = [])

def test_trash():
    with pytest.raises(AssertionError):
        v = VacuumLand(trash = False)
    with pytest.raises(AssertionError):
        v = VacuumLand(trash = 10.1)

def test_pytest():
    with pytest.raises(AssertionError):
        assert 1 == 2
