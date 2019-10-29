#/bin/bash
# 
# Local tests of CI jobs
# Run this from CI job docker container
set -ex

echo 'Running black'
black --check orbyter_demo

echo 'Running flake'
flake8 orbyter_demo

echo 'Running pytest'
pytest orbyter_demo

echo 'Finished tests'
