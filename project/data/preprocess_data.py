from stop_words import get_stop_words
from pyarabic.araby import strip_harakat
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import contractions
import nltk
nltk.download('wordnet')
stop_words_ar = get_stop_words('arabic')
stop_words_en = get_stop_words('english')


class Preprocess:
    def __init__(self, df_data):
        self.df_data = df_data

    def __call__(self):
        self.label_data()
        self.df_data['text_processed'] = self.df_data.apply(lambda x: self.clean(x['text'], x['language']), axis=1)
        return self.df_data[['text_processed', 'target']]

    def label_data(self):
        self.df_data['popularity_index'] = (self.df_data['retweet_count'] + 1) * (
        (self.df_data['favorite_count'] + 1)) / (.1 * (self.df_data['follower_count'] + 1))
        self.df_data['target'] = self.df_data['popularity_index'].map(
            lambda x: 0 if x < 0.1 else (1 if x >= 0.1 and x < 10 else 2))

    @staticmethod
    def clean(text, language='en'):
        text = re.sub("@[A-Za-z0-9]+", '', text)
        text = re.sub(r'#\w+ ?', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'RT', '', text)

        if language == 'en':
            text = text.lower()
            text = ' '.join([word for word in text.split() if word not in (stop_words_en)])
            text = ' '.join(WordNetLemmatizer().lemmatize(word) for word in text.split())
            text = ' '.join(PorterStemmer().stem(word) for word in text.split())
            text = ' '.join([contractions.fix(word) for word in text.split()])

        else:
            text = re.sub(r'\s*[A-Za-z]+\b', '' , text)
            text = text.rstrip()
            text = strip_harakat(text)
            text = ' '.join([word for word in text.split() if word not in (stop_words_ar)])

        return text



