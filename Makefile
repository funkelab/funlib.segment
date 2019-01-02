default:
	pip install -r requirements.txt
	pip install .

.PHONY: test
test:
	python -m tests -v
