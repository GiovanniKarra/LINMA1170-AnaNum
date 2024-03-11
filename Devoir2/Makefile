SUBMISSION_FILES = devoir2.py rapport/rapport.pdf rapport/rapport.tex rapport/images
LATEX_FILES = rapport.aux rapport.fdb_latexmk rapport.fls rapport.log rapport.pdf rapport.synctex.gz
LATEX_DIR = svg-inkscape

submit: pdf
	@zip devoir2.zip $(SUBMISSION_FILES)

pdf:
	@cd rapport; pdflatex --shell-escape rapport.tex

clean:
	@rm -f devoir2.zip

	@cd rapport; rm -f $(LATEX_FILES); rm -rf $(LATEX_DIR)