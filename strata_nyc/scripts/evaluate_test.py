import pytest
from click.testing import CliRunner

from strata_nyc.scripts.evaluate import evaluate


@pytest.mark.parametrize("config_file", [("/mnt/configs/test_config.yml")])
def test_evaluate(config_file):
    runner = CliRunner()
    result = runner.invoke(evaluate, [config_file])
    assert result.exit_code == 0
