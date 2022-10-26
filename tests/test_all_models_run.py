from gms.common import discover_models


def test_models():
    models = discover_models()
    print(models)
