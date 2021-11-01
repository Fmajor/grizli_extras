all:
	@cat makefile
debug:
	which python
install:
	python setup.py develop
uninstall:
	python setup.py develop --uninstall
