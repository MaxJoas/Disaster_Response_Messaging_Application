k.PHONY: create_environment git

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = disaster_messaging_classification_model
CONDA_ENVIRONMENT = disaster_messaging_classification_model
PYTHON_VERSION = Python interpreter (e.g. 3, 3.5, 2.7)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Poetry dependencies
poetry-install:
	poetry lock
	poetry config virtualenvs.create false \
		&& poetry export --without-hashes -f requirements.txt --dev\
		|  pip install -r /dev/stdin \
		&& poetry debug

build-setup:
	pip install dephell
	dephell deps convert

nltk-resources:
	python -m nltk.downloader punkt 
	python -m nltk.downloader stopwords
	python -m nltk.downloader wordnet  

process-data:
	PYTHONPATH=.. python ../disaster_messaging_classification_model/utils/data_utils.py

# train model
train-model:
	PYTHONPATH=.. python ../disaster_messaging_classification_model/model/train_model.py

## test price prediction model
test-model:
	pip install pytest==6.1.1
	PYTHONPATH=. python -m pytest -v tests 

package:
	pip install --user twine
	python setup.py sdist
	twine upload --skip-existing dist/*

