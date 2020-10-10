import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# import tokenize_function
from models.tokenizer_function import Tokenizer

# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from disaster_messaging_classification_model.config import config


def load_data(database_filepath):
    """
        Load data from the sqlite database. 
    Args: 
        database_filepath: the path of the database file
    Returns: 
        X (DataFrame): messages 
        Y (DataFrame): One-hot encoded categories
        category_names (List)
    """

    # load data from database
    engine = create_engine(f"sqlite:///data/{config.DATABASE_NAME}")
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df[config.MESSAGE_FEATURE]
    Y = df.drop(config.EXTRA_FEATURES_DROP_Y, axis=1)
    category_names = Y.columns

    return X, Y, category_names


def build_model():
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
            (
                "clf",
                MultiOutputClassifier(
                    AdaBoostClassifier(n_estimators=config.N_ESTIMATORS)
                ),
            ),
        ]
    )

    # grid search

    cv = GridSearchCV(
        estimator=pipeline,
        param_grid=config.GRID_CV_PARAMETERS,
        cv=config.CV_FOLDS,
        n_jobs=config.N_JOBS,
    )

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluate the model performances, in terms of f1-score, precison and recall
    Args: 
        model: the model to be evaluated
        X_test: X_test dataframe
        Y_test: Y_test dataframe
        category_names: category names list defined in load data
    Returns: 
        perfomances (DataFrame)
    """
    # predict on the X_test
    y_pred = model.predict(X_test)

    # build classification report on every column
    performances = []
    for i in range(len(category_names)):
        performances.append(
            [
                f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average="micro"),
                precision_score(
                    Y_test.iloc[:, i].values, y_pred[:, i], average="micro"
                ),
                recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average="micro"),
            ]
        )
    # build dataframe
    performances = pd.DataFrame(
        performances, columns=["f1 score", "precision", "recall"], index=category_names
    )
    return performances


def save_model(model, model_filepath):
    """
        Save model to pickle
    """
    joblib.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")


if __name__ == "__main__":
    main()
