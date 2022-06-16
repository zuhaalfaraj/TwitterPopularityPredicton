import configparser

config = configparser.ConfigParser()

config['tokenizer_param'] = {"add_special_tokens": True, "padding": True, "truncation": True, 'max_length': 200}
config['model_checkpoint'] ='bert-base-multilingual-cased'
config['TRAIN_BATCH_SIZE'] = 16
config['VALID_BATCH_SIZE'] = 16
config['TEST_BATCH_SIZE'] = 16
config['classes_num'] = 3
config['learning_rate'] = 1e-5
config['random_state'] = 101
config['splitting_ratio'] = 0.2
config['s3_data_dir'] = 'preprocessed_data/twitter_data.csv'


