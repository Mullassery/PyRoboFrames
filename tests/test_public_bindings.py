import pyroboframes


def test_native_loader_and_validation_types_are_exported():
    assert pyroboframes.PrefetchLoader is not None
    assert pyroboframes.ValidationReport is not None
