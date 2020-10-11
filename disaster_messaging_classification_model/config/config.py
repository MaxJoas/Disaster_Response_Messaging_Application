import pathlib
import pandas as pd
import disaster_messaging_classification_model

pd.options.display.max_rows = 100
pd.options.display.max_columns = 50

PACKAGE_ROOT = (
    pathlib.Path(disaster_messaging_classification_model.__file__).resolve().parent
)

MODEL_NAME = "adaboost"
MODEL_SAVE_FILE = f"{MODEL_NAME}_model_v"

TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
PERFORMACE_REPORT_DIR = "performance_report"
DATASET_DIR = PACKAGE_ROOT / "data"
DATA_FILE_NAME = "disaster_messages.csv"
DATA_CAT_FILE_NAME = "disaster_categories.csv"
DATABASE_NAME = "DisasterResponse.db"
TABLE_NAME = "disaster_messages"

EXTRA_FEATURES_DROP_Y = ["id", "message", "original", "genre"]
MESSAGE_FEATURE = "message"

PARAMS = {"n_estimators": 150}
RANDOM_SEED = 7

CV_FOLDS = 5
N_JOBS = 10

N_ESTIMATORS = 100

