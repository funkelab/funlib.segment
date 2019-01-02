default:
	pip install -r requirements.txt
	pip install .

.PHONY: tests
tests:
	pip install -r requirements_dev.txt
	pytest -v funlib
