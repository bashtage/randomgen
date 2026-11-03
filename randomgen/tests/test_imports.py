import importlib
import importlib.util
import pathlib
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


# Internal helper without a public API
NO_STUB = ("randomgen.broadcasting",)
EXTENSIONS = [
    mod
    for mod in MODULES
    if mod not in NO_STUB
    and not importlib.util.find_spec(mod).origin.endswith((".py", ".pyc"))
]


def test_found_extensions():
    assert "randomgen.squares" in EXTENSIONS
    assert len(EXTENSIONS) > 20


@pytest.mark.parametrize("module", EXTENSIONS)
def test_stub_installed(module):
    origin = pathlib.Path(importlib.util.find_spec(module).origin)
    stub = origin.with_name(module.rsplit(".", maxsplit=1)[-1] + ".pyi")
    assert stub.exists()
