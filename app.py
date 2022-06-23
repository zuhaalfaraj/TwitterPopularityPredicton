from flask import Flask
from flask_restful import Api
from resources.tweet import TweetResource
from project.training.model import BERTRegressor
from flask import request


app = Flask(__name__)
app.secret_key = 'Zuuha'
api = Api(app)


api.add_resource(TweetResource, '/assessment')


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return {'message':'Server shutting down...'}


if __name__ == '__main__':
    app.run(port=5000, debug=True)