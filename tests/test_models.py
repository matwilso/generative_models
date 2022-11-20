import subprocess

import pytest

from gms import common

models = sorted(common.discover_models().keys())


@pytest.mark.parametrize("model_name", models)
def test_all_models(model_name):
    cmd = f'python -m gms.main --model={model_name} --epochs=1 --logdir logs/pytest/{model_name}'
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
