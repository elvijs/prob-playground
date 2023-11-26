# A collection of convenience tools for running the typical tasks


install_deps:
	pip install -r requirements.txt


qa: format static_checks


static_checks: lint type_check

type_check:
	mypy .

lint:
	ruff check .


lint_fix:
	ruff check --fix .


format:
	ruff format .


freeze_requirements:
	pip freeze | grep -v personal_finance.git > requirements.txt
