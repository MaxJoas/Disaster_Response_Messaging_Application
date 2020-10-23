import math

from disaster_messaging_classification_model.model.predict import make_prediction
from disaster_messaging_classification_model.utils.model_utils import load_data_from_db


def test_make_single_prediction():
    # Given
    X_test, y_test, categories = load_data_from_db(set_label="test")
    single_test_input = X_test[0:1]

    # When
    subject = make_prediction(input_data=single_test_input)

    # Then
    assert subject is not None
    assert isinstance(subject.get("predictions")[0], dict)
    assert len(subject.get("predictions")[0]) == 37
    assert subject.get("predictions")[0]["related"] == 1
    assert subject.get("predictions")[0]["weather_related"] == 1
    assert subject.get("predictions")[0]["storm"] == 1

    for i in categories:
        if i not in ["related", "weather_related", "storm"]:
            assert subject.get("predictions")[0][i] == 0

