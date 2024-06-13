import math

from ..calculate import avg


def test_avg():
    result = avg(0.1, 0.7, 2, 1)
    expected = 0.3
    assert math.isclose(result, expected)
