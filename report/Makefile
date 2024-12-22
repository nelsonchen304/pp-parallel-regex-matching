.PHONY: all clean

REPORT=pp-final

all: $(REPORT).pdf

$(REPORT).pdf: $(REPORT).tex $(REPORT).bib
	latexmk -pdf $(REPORT).tex

clean:
	latexmk -C
	rm $(REPORT).bbl