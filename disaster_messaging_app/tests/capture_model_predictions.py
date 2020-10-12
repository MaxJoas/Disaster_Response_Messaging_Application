"""
This script should only be run in CI.
Never run it locally or you will disrupt the
differential test versioning logic.
"""

import pandas as pd

from disaster_messaging_classification_model.model.predict import make_prediction
from disaster_messaging_classification_model.utils.model_utils import load_data_from_db

from api import config


def capture_predictions() -> None:
    """Save the test data predictions to a CSV."""

    save_file = "test_data_predictions.csv"
    X_test, y_test, categories = load_data_from_db(set_label="test")

    # we take a slice with no input validation issues
    multiple_test_input = X_test[99:600]

    predictions = make_prediction(input_data=multiple_test_input)

    # save predictions for the test dataset
    predictions_df = pd.DataFrame(predictions)

    # hack here to save the file to the regression model
    # package of the repo, not the installed package
    predictions_df.to_csv(f"{config.PACKAGE_ROOT}/{save_file}")


if __name__ == "__main__":
    capture_predictions()
