import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class TwitterAPIDataCollection:
    def __init__(self):
        auth = tweepy.OAuth1UserHandler(os.environ['API_KEY'], os.environ['API_SECRET_KEY'])
        auth.set_access_token(os.environ['ACCESS_TOKEN'], os.environ['ACCESS_TOKEN_SECRET'])
        self.api = tweepy.API(auth)
    @staticmethod
    def get_all_users(table):
        users = table['user_id'].unique()
        return list(users)

    def get_data(self):
        dataset = []
        visited_accounts = []
        for tweet in tweepy.Cursor(self.api.home_timeline, count=300).items():

            entitiy = {}
            entitiy['tweet_id'] = tweet.id_str
            entitiy['text'] = tweet.text
            entitiy['user_id'] = tweet.author.id_str
            entitiy['user_name'] = tweet.author.name
            entitiy['status_count'] = tweet.author.statuses_count
            entitiy['follower_count'] = tweet.author.followers_count
            entitiy['friends_count'] = tweet.author.friends_count
            entitiy['profile_description'] = tweet.author.description
            entitiy['verified_user'] = tweet.author.verified
            entitiy['tweet_date'] = tweet.created_at
            entitiy['language'] = tweet.lang
            entitiy['favorite_count'] = tweet.favorite_count
            entitiy['retweet_count'] = tweet.retweet_count

            dataset.append(entitiy)

            if entitiy['user_id'] not in visited_accounts:
                visited_accounts.append(entitiy['user_id'])

                for user_tweet in self.api.user_timeline(user_id=tweet.author.id_str, count=200):
                    entitiy = {}

                    entitiy['tweet_id'] = user_tweet.id_str
                    entitiy['text'] = user_tweet.text
                    entitiy['user_id'] = user_tweet.author.id_str
                    entitiy['user_name'] = user_tweet.author.name
                    entitiy['status_count'] = user_tweet.author.statuses_count
                    entitiy['follower_count'] = user_tweet.author.followers_count
                    entitiy['friends_count'] = user_tweet.author.friends_count
                    entitiy['profile_description'] = user_tweet.author.description
                    entitiy['verified_user'] = user_tweet.author.verified
                    entitiy['tweet_date'] = user_tweet.created_at
                    entitiy['language'] = user_tweet.lang
                    entitiy['favorite_count'] = user_tweet.favorite_count
                    entitiy['retweet_count'] = user_tweet.retweet_count

                    dataset.append(entitiy)

        data_df = pd.DataFrame(dataset)
        return data_df

