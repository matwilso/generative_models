
test: format-check lint

lint:
	find . -type f -name "*.py" | xargs pylint

format-check:
	black -S --line-length 90 --check .
	isort --profile black --check-only .

format:
	black -S --line-length 90 .
	isort --profile black .

test-run-models:
	 python -m pytest -v -x -s
