.PHONY: requirements
requirements:
	pip-compile --output-file requirements.txt --quiet requirements.in
	pip-sync requirements.txt
.PHONY: typehint
typehint:
	mypy --config-file pyproject.toml app/
.PHONY: clean
clean:
	find . -type d -name .mypy_cache | xargs rm -fr
.PHONY: black
black:
	black app/ tests/