
test: format-check lint

test-heavy: format-check lint test-run-models

lint:
	find . -type f -name "*.py" | xargs pylint

format-check:
	black -S --line-length 90 --check .
	isort --profile black --check-only .

format:
	black -S --line-length 90 .
	isort --profile black .

test-run-models:
	 pytest -v -x
