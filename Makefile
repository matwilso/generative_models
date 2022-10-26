
test: format-check lint

lint:
	find . -type f -name "*.py" | xargs pylint

format-check:
	black -S --check .
	isort --profile black --check-only .

format:
	black -S .
	isort --profile black .

test-run-models:
	 python -m pytest -v -x -s
