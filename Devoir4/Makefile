SUBMISSION_ARCHIVE = devoir4.zip
SUBMISSION_FILES = code/*\
				   rapport/rapport.pdf rapport/rapport.tex rapport/images
LATEX_FILES = rapport.aux rapport.fdb_latexmk rapport.fls rapport.log rapport.pdf\
			  rapport.synctex.gz rapport.out
LATEX_DIR = svg-inkscape

submit: pdf
	@zip $(SUBMISSION_ARCHIVE) $(SUBMISSION_FILES)

pdf:
	@cd rapport; pdflatex --shell-escape rapport.tex

clean:
	@rm -f $(SUBMISSION_ARCHIVE)

	@cd rapport; rm -f $(LATEX_FILES); rm -rf $(LATEX_DIR)