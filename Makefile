SHELL := /bin/bash

build:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python setup.py test

lint:
	flake8 ./auditai

coverage:
	coverage run --source auditai setup.py test
	coverage report

covhtml:
	coverage html
	open ./htmlcov/index.html

clean:
	@rm -Rf *.egg-info .eggs .tox build dist htmlcov .coverage
	@find ./ -depth -type d -name __pycache__ -exec rm -Rf {} \;
	@find ./ -type f \( -iname \*.pyc -o -iname \*.pyo -o -iname \*~ \) -delete

test-all: coverage lint

pubdev: clean
	python setup.py bdist_wheel
	twine upload --repository testpypi dist/*

publish: clean
	python setup.py bdist_wheel
	twine upload --repository pypi dist/*


.PHONY: build dev test lint coverage covhtml clean test-all pubdev publish