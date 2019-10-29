import pytest
from click.testing import CliRunner

from orbyter_demo.scripts.predict import predict


@pytest.mark.parametrize("config_file", [("configs/test_config.yml")])
def test_predict(config_file):
    runner = CliRunner()
    result = runner.invoke(predict, [config_file])
    assert result.exit_code == 0
