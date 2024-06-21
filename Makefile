default: install

.PHONY: install
install:
	pip install -r requirements.txt
	pip install .

.PHONY: install-dev
install-dev:
	pip install -r requirements_dev.txt
	pip install -e .

.PHONY: tests
tests:
	pytest -v --cov=funlib funlib
	flake8 funlib
