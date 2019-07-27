#!/bin/bash

if [ -d "fastText" ]; then
  echo "FastText folder already exists. Delete it"
else
  pip install --user --upgrade pip
  pip install --user pybind11
  git clone https://github.com/facebookresearch/fastText.git
  cd fastText
  pip install --user .
fi
