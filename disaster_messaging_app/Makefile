#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = disaster_reponse_app
CONDA_ENVIRONMENT = disaster_reponse_app
PYTHON_VERSION = Python interpreter (e.g. 3, 3.5, 2.7)
NAME=disaster-response-app184
HEROKU_APP_NAME=disaster-response-app184
COMMIT_ID=$(shell git rev-parse HEAD)  # REMOVED

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Poetry dependencies
poetry-install:
	poetry lock
	poetry config virtualenvs.create false \
		&& poetry export --without-hashes -f requirements.txt --dev\
		|  pip install -r /dev/stdin  --use-feature=2020-resolver\
		&& poetry debug

## test model api
test-app:
	pip install pytest
	PYTHONPATH=. python -m pytest -v -m "not differential" tests

build-app-heroku:
	poetry export -f requirements.txt -o requirements.txt
	docker build -t registry.heroku.com/$(NAME)/web .

push-app-heroku:
	docker push registry.heroku.com/${HEROKU_APP_NAME}/web:latest
