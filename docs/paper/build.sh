#!/bin/bash
set -e
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex 