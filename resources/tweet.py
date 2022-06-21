from flask_restful import Resource, reqparse
from flask_jwt import jwt_required
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
        out = TweetModel(**data)
        if out:
            return {'Assessment result': float(out)}
        return {"message": "not found"}, 404
