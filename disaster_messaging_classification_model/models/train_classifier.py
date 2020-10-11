import sys
import pandas as pd
import numpy as np
import pickle
import logging

# import tokenize_function
from disaster_messaging_classification_model.features.message_tokenizer import Tokenizer

# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier

from disaster_messaging_classification_model.config import config
from disaster_messaging_classification_model.utils.model_utils import (
    load_data_from_db,
    evaluate_model,
    save_pipeline,
)
from disaster_messaging_classification_model import __version__ as _version


def build_model_pipeline():
    """
      build NLP pipeline - count words, tf-idf, multiple output classifier,
      grid search the best parameters
    Args: 
        None
    Returns: 
        cross validated classifier object
    """
    #
    pipeline = Pipeline(
        [
            ("tokenizer", Tokenizer()),
            ("vec", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(AdaBoostClassifier(**config.PARAMS))),
        ]
    )
    return pipeline


def train_model():

    _logger = logging.getLogger(__name__)

    database_filepath = config.DATASET_DIR / config.DATABASE_NAME
    model_filepath = config.TRAINED_MODEL_DIR / config.MODEL_SAVE_FILE

    _logger.info("Loading data...\n    DATABASE: {}".format(database_filepath))

    # load and split data
    X, Y, category_names = load_data_from_db(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # build pipeline
    _logger.info("Building model...")
    model = build_model_pipeline()

    # train model
    _logger.info("Training model...")
    model.fit(X_train, Y_train)

    # evaluate model
    _logger.info("Evaluating model...")
    evaluate_model(model, X_test, Y_test, category_names)

    # save model pipelin
    _logger.info(f"Saving model...")
    save_pipeline(pipeline_to_persist=model)

    _logger.info("Trained model saved!")


if __name__ == "__main__":
    train_model()
