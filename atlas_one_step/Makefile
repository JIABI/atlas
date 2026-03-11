
.PHONY: install test smoke lint
install:
	pip install -e .

test:
	pytest -q

smoke:
	python -m atlas_one_step.cli.smoke_test
