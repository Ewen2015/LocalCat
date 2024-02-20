#!/bin/bash

rm dist/*
python3 -m build --wheel
twine upload dist/*
rm dist/*
