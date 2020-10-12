from flask import Blueprint, request, jsonify, render_template
from disaster_messaging_classification_model.model.predict import make_prediction
from disaster_messaging_classification_model import __version__ as _version
from disaster_messaging_classification_model.visualizations.generate_visuals import (
    VisualsGeneration,
)
import os
from werkzeug.utils import secure_filename

from app.config import get_logger, UPLOAD_FOLDER

# from app.validation import validate_inputs
from app import __version__ as app_version
import json
import plotly

_logger = get_logger(logger_name=__name__)


classification_app = Blueprint("classification_app", __name__)


@classification_app.route("/")
@classification_app.route("/index")
def index():
    graphs = VisualsGeneration().generate_plotly_word_cloud_visuals()
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


@classification_app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        _logger.info("health status OK")
        return "ok"


@classification_app.route("/version", methods=["GET"])
def version():
    if request.method == "GET":
        return jsonify({"model_version": _version, "app_version": app_version})


@classification_app.route("/v1/predict/model", methods=["POST"])
def predict():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f"Inputs: {json_data}")

        # Step 3: Model prediction
        response = make_prediction(input_data=json_data)
        _logger.debug(f"Outputs: {response}")

        # Step 4: Convert numpy ndarray to list
        predictions = response.get("predictions")
        version = response.get("version")

        # Step 5: Return the response as JSON
        return jsonify({"predictions": predictions, "version": version})


@classification_app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    response = make_prediction(input_data=query)
    _logger.debug(f"Outputs: {response}")

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=response["predictions"][0]
    )

