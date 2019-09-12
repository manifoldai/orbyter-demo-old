#/bin/bash
# 
# Local tests of CI jobs
# Run this from CI job docker container
set -ex

echo 'Running black'
black --check strata_nyc

echo 'Running flake'
flake8 strata_nyc

echo 'Running pytest'
pytest strata_nyc

echo 'Finished tests'
