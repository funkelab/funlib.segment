default:
	pip install .

.PHONY: test
test:
	python -m tests -v
