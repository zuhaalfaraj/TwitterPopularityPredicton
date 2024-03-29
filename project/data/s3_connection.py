import boto3
import io
import pandas as pd
import os
import torch
from config import config
from dotenv import load_dotenv
from project.training.model import BERTRegressor
load_dotenv()


class S3Connection:
    def __init__(self, bucket_name="twitter-viral-project"):
        session = boto3.Session(
            aws_access_key_id= os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key= os.environ["AWS_SECRET_ACCESS_KEY"])

        self.s3 = session.resource('s3')
        self.client = session.client('s3')

        self.bucket = self.s3.Bucket(bucket_name)
        self.bucket_name = bucket_name

    def upload_data(self, data_dir, local_dir=None, df=None):
        if df is not None:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            self.client.put_object(Body=csv_buffer.getvalue(), Bucket=self.bucket_name, Key=data_dir)

        elif data_dir:
            self.bucket.upload_file(local_dir, data_dir)

    def upload_model(self, model_dir, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        self.client.put_object(Bucket=self.bucket_name, Key=model_dir, Body=buffer.getvalue())

    def read_model(self, model_dir, model):
        mod = self.client.get_object(Bucket=self.bucket_name, Key=model_dir)['Body'].read()
        buffer = io.BytesIO(mod)
        model.load_state_dict(torch.load(buffer, map_location=config['device']))
        return model

    def update_data(self, data_dir, new_records_df):
        old_data = self.client.get_object(Bucket=self.bucket_name, Key=data_dir)['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(old_data))
        updated_df = pd.concat([df, new_records_df]).drop_duplicates()

        csv_buffer = io.StringIO()
        updated_df.to_csv(csv_buffer, index=False)
        self.client.put_object(Body=csv_buffer.getvalue(), Bucket=self.bucket_name, Key=data_dir)

        return updated_df

    def read_data(self, data_dir):
        old_data = self.client.get_object(Bucket=self.bucket_name, Key=data_dir)['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(old_data))
        return df


    def delete_data(self, data_dir):
        self.client.delete_object(Bucket=self.bucket_name, Key= data_dir)
