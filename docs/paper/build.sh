#!/bin/bash
set -e
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
rm *.aux *.log *.bbl *.blg *.out *.fls *.fdb_latexmk
mv main.pdf tatbot2025.pdf