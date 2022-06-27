# TwitterPopularityPredicton

## An Overview
The overall purpose of this project is to build a complete machine learning pipeline that contains:
- Data collection
- Data cleaning
- Data Preprocessing
- Model Training
- Testing and Deployment


The use case here is to predict how popular the tweet will be based on the written text. This would help any marketing agency or influencers to increase the reach of their content.

## Files
- ``` /project```
  - ``` /data ```
     - ``` api_data_collection.py ```: the class that was built to collect data from twitter.
     -  ``` preprocess_data.py ```: the class that was built to preprocess text data based on the language.
     -  ``` s3_connection.py ```: the class that was built to read and write files to S3.
     -  ``` s3_connection.py ```: the class that was built to split data to train/test/val.
  - ``` /training ```
    - ``` dataset_class.py ```: The dataset class for Dataloader.
    - ``` evaluation.py ```: The evaluation metrics.
    - ``` model.py ```: The corresponding model.
    - ``` training_loop.py ```: The training loop (Parent: normal trianing loop, child: Training loop that logs results to wandb)
    - ``` training_pipeline.py ```: The full trianing pipeline

- ``` /models```: API related models
- ``` /resources```: API related resources

## Run 
 Run Docker image
 
```
docker run zuhaalfaraj/twitter_viral
```

## Use the API
- Send a GET request to the following path: ```{host_name}/assessment``` 
- Send the body with the following structure:
```
{
    "tweet": "Hello World.. This is a test" ,
    "lang": "en"
}
```

