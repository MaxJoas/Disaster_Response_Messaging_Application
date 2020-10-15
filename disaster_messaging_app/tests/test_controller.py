import io
import json
import math
import os

from disaster_messaging_classification_model import __version__ as _version
from disaster_messaging_classification_model.config import config as model_config
from disaster_messaging_classification_model.utils.model_utils import load_data_from_db

from app import __version__ as app_version


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get("/health")

    # Then
    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    # When
    response = flask_test_client.get("/version")

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json["model_version"] == _version
    assert response_json["app_version"] == app_version


def test_prediction_endpoint_populates_front_end(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    X_test, y_test, categories = load_data_from_db(set_label="test")
    post_json = X_test[0:1].to_json(orient="records")

    # When
    response = flask_test_client.get("/go", json=json.loads(post_json))
    # Then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    X_test, y_test, categories = load_data_from_db(set_label="test")
    post_json = X_test[0:1].to_json(orient="records")

    # When
    response = flask_test_client.post("/v1/predict/model", json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)

    prediction = response_json["predictions"]
    response_version = response_json["version"]
    assert response_json is not None
    assert isinstance(response_json.get("predictions")[0], dict)
    assert len(response_json.get("predictions")[0]) == 36
    assert response_json.get("predictions")[0]["related"] == 1

    for i in categories:
        if i != "related":
            assert response_json.get("predictions")[0][i] == 0
    assert response_version == _version
