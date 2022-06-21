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

    def get(self):
        data = TweetResource.parse.parse_args()
        out = TweetModel(*data).get_assessment()
        if out:
            return {'Assessment result': float(out)}
        return {"message": "not found"}, 404
