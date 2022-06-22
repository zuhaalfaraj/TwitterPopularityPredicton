from flask_restful import Resource, reqparse
from models.tweets import TweetModel
from project.training.model import BERTRegressor
from project.data.s3_connection import S3Connection
from config import config


model = S3Connection().read_model(config['s3_model_dir'], BERTRegressor())
tweet_model = TweetModel()


class TweetResource(Resource):
    parse = reqparse.RequestParser()
    parse.add_argument('tweet',
                       type=str,
                       required=True,
                       help="This field cannot left blank")
    parse.add_argument('lang',
                       type=str,
                       required=True,
                       help="This field cannot left blank")

    def get(self):
        data = TweetResource.parse.parse_args()
        out = tweet_model.get_assessment(*data, model= model)
        if out:
            return {'Assessment result': float(out)}
        return {"message": "not found"}, 404
