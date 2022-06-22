from flask_restful import Resource, reqparse
from models.tweets import TweetModel


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

    def __init__(self):
        self.tweet_model = TweetModel()

    def get(self):
        data = TweetResource.parse.parse_args()
        out = self.tweet_model.get_assessment(*data)
        if out:
            return {'Assessment result': float(out)}
        return {"message": "not found"}, 404
