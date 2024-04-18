#!/bin/bash

# Use the provided file parameter to run the experiment
python -O experiments.py -b $HOME/results -j $1 -c
