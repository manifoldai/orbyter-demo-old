#/bin/bash
#
# autoformat.sh
# 
# Runs all autoformaters
# Run this from CI job docker container
set -ex

echo 'Running isort'
isort -rc orbyter_demo

echo 'Running black'
black orbyter_demo

echo 'Finished auto formatting'
