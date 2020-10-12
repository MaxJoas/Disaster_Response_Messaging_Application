from flask import Flask

from api.config import get_logger


_logger = get_logger(logger_name=__name__)


def create_app(*, config_object) -> Flask:
    """Create a flask app instance."""

    flask_app = Flask("disaster_messaging_app")
    flask_app.config.from_object(config_object)

    # import blueprints
    from api.controller import classification_app

    flask_app.register_blueprint(classification_app)
    _logger.debug("Application instance created")

    return flask_app
