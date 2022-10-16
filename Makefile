
test:
	black -S --check .
	isort --profile black --check-only .

format:
	black -S .
	isort --profile black .
