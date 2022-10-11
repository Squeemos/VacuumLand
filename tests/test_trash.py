import pytest
from vacuum_land import VacuumLand

def test_trash_bool():
    with pytest.raises(AssertionError):
        v = VacuumLand(trash = False)

def test_trash_float():
    with pytest.raises(AssertionError):
        v = VacuumLand(trash = 10.1)

def test_trash_list():
    with pytest.raises(AssertionError):
        v = VacuumLand(trash = [])

def test_trash_zero():
    with pytest.raises(AssertionError):
        v = VacuumLand(trash = 0)

def test_trash_big():
    with pytest.raises(AssertionError):
        v = VacuumLand(width = 1, height = 1, trash = 2)
