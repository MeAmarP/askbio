import pytest

import sys
sys.path.append('../src')


from main import hworld

def test_helloworld():
    """
    JUST A HELLO WORLD TEST
    """

    assert hworld() == "Hello World"
