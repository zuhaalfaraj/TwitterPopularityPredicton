from flask import Flask
from flask_restful import Api
from resources.tweet import TweetResource
from project.training.model import BERTRegressor


app = Flask(__name__)
app.secret_key = 'Zuuha'
api = Api(app)


api.add_resource(TweetResource, '/assessment')


if __name__ == '__main__':
    app.run(port=5000, debug=True)