default:
	pip install -r requirements.txt
	pip install .

.PHONY: tests
tests:
	pytest -v --cov=funlib funlib
	flake8 funlib
