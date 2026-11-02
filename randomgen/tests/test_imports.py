import importlib
import pkgutil

import pytest

import randomgen

SKIP = ("randomgen.tests", "randomgen.examples", "randomgen._build")
MODULES = [
    mod.name
    for mod in pkgutil.walk_packages(randomgen.__path__, "randomgen.")
    if not mod.name.startswith(SKIP)
]


def test_found_modules():
    assert "randomgen.generator" in MODULES
    assert len(MODULES) > 20


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)
