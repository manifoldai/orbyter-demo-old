#/bin/bash
#
# autoformat.sh
# 
# Runs all autoformaters
# Run this from CI job docker container
set -ex

echo 'Running isort'
isort -rc strata_nyc

echo 'Running black'
black strata_nyc

echo 'Finished auto formatting'
