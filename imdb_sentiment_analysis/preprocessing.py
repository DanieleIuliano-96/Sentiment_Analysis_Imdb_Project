# preprocessing.py
import re
import nltk
import pandas as pd
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')


def read_data(file_path):
    return pd.read_csv(file_path)


def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)


def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem


def to_lower(text):
    return text.lower()


def rem_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = nltk.word_tokenize(text)
    return ' '.join([w for w in words if w.lower() not in stop_words])


def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text.split()])


def preprocess_data(data):
    data['review'] = data['review'].apply(clean)
    data['review'] = data['review'].apply(is_special)
    data['review'] = data['review'].apply(to_lower)
    data['review'] = data['review'].apply(rem_stopwords)
    data['review'] = data['review'].apply(stem_txt)
    return data


