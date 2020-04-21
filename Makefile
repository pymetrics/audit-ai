SHELL := /bin/bash

build:
	pip install -e .

install-dev: clean
	pip install -e ".[dev]"

tests:
	pytest --cov-config=setup.cfg --cov=auditai --cov-fail-under=65

lint:
	flake8 ./auditai

covhtml:
	coverage html
	open ./htmlcov/index.html

clean:
	@rm -Rf *.egg-info .eggs .tox build dist htmlcov .coverage
	@find ./ -depth -type d -name __pycache__ -exec rm -Rf {} \;
	@find ./ -type f \( -iname \*.pyc -o -iname \*.pyo -o -iname \*~ \) -delete

test-all: tests lint

pubdev: is_newest_version clean
	python setup.py bdist_wheel
	twine upload --repository testpypi dist/*

is_newest_version:
	python auditai/is_newest_version.py

publish: is_newest_version clean
	python setup.py bdist_wheel
	twine upload --repository pypi dist/*

tox_tests:
	tox

.PHONY: build install-dev tests lint coverage covhtml clean test-all pubdev publish