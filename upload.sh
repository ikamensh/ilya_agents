#!/usr/bin/env bash

rm -r dist
python setup.py sdist bdist_wheel
python -m twine upload -r testpypi dist/* -u ikamensh