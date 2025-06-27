import importlib
def test_script_imports():
    assert importlib.import_module("analysis_Rotation") is not None
