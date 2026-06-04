PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all assets clean

all: assets main.pdf

assets:
	cd .. && $(PYTHON) experiments/generate_conference_report_assets.py

main.pdf: main.tex references.bib
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
