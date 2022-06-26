from flask import Flask
from flask_restful import Api
from resources.tweet import TweetResource
from project.training.model import BERTRegressor
from flask import request


def create_app():
    app = Flask(__name__)
    app.secret_key = 'Zuuha'
    api = Api(app)
    api.add_resource(TweetResource, '/assessment')

    return app


if __name__ == '__main__':
    create_app().run(debug=True)
