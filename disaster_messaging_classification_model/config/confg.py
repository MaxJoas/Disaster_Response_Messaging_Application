import pathlib
import pandas as pd
import disaster_messaging_classification_model

pd.options.display.max_rows = 100
pd.options.display.max_columns = 50

PACKAGE_ROOT = (
    pathlib.Path(disaster_messaging_classification_model.__file__).resolve().parent
)
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_model_files"
DATASET_DIR = PACKAGE_ROOT / "data"
DATA_FILE_NAME = "disaster_messages.csv"
DATA_CAT_FILE_NAME = "disaster_categories.csv"
DATABASE_NAME = "DisasterResponse.db"

EXTRA_FEATURES_DROP_Y = ["id", "message", "original", "genre"]
MESSAGE_FEATURE = "message"

GRID_CV_PARAMETERS = {
    "clf__estimator__max_features": ["sqrt", 0.5],
    "clf__estimator__n_estimators": [50, 100],
}

CV_FOLDS = 5
N_JOBS = 10

N_ESTIMATORS = 100
