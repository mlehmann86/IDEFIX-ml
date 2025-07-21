#!/bin/zsh

# Find and remove .png files
find . -type f -name '*.png' -exec rm -f {} \;

# Find and remove .vtk files
find . -type f -name '*.vtk' -exec rm -f {} \;

find . -type f -name '*.log' -exec rm -f {} \;

find . -type f -name '*.dat' -exec rm -f {} \;
