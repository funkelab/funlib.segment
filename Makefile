default:
	pip install .
	-rm -rf dist build segment.egg-info

.PHONY: test
test:
	python -m tests -v
