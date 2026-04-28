import pytest


_MISSING_GLUON_IR_REASON = (
    "Gluon tests require the triton._C.libtriton.gluon_ir native binding")


def _has_gluon_ir_binding():
    try:
        from triton._C.libtriton import gluon_ir  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_GLUON_IR_BINDING = _has_gluon_ir_binding()


class _UnavailableGluonModule(pytest.Module):

    def collect(self):
        yield _UnavailableGluonItem.from_parent(self, name="gluon_ir_binding")


class _UnavailableGluonItem(pytest.Item):

    def runtest(self):
        pytest.skip(_MISSING_GLUON_IR_REASON)

    def reportinfo(self):
        return self.path, 0, _MISSING_GLUON_IR_REASON


def pytest_pycollect_makemodule(module_path, parent):
    if _HAS_GLUON_IR_BINDING:
        return None

    if module_path.name.startswith("test_"):
        return _UnavailableGluonModule.from_parent(parent, path=module_path)

    return None
