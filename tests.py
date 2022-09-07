import pytest
from vacuum_land import VacuumLand

def test_height():
    with pytest.raises(AssertionError):
        v = VacuumLand(height = False)
        v = VacuumLand(height = 10.1)
        v = VacuumLand(height = [])
