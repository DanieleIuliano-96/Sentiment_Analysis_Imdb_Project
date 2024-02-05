import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Rimani nella stessa sessione Python per garantire che le risorse NLTK siano già scaricate
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Rimani nella stessa sessione Python per garantire che le risorse NLTK siano già scaricate
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk import bigrams


def preprocess_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


def top_words_for_sentiment(reviews, sentiments, sentiment_type='bad', top_n=10):
    sentiment_reviews = [review for review, sentiment in zip(reviews, sentiments)
                         if sentiment == 'negative'] if sentiment_type == 'bad' else \
        [review for review, sentiment in zip(reviews, sentiments) if sentiment == 'positive']

    stopwords = {'prot', 'gunga', 'prue', 'gypo', 'yokai', 'existenz'}

    all_words = [word for review in sentiment_reviews for word in preprocess_text(review) if word not in stopwords]
    word_counts = Counter(all_words)

    bigram_counts = Counter(list(bigrams(all_words)))

    top_words = [(word, count) for word, count in word_counts.items()]
    top_words = sorted(top_words, key=lambda x: x[1], reverse=True)[:top_n]

    top_bigrams = [(bigram, count) for bigram, count in bigram_counts.items()]
    top_bigrams = sorted(top_bigrams, key=lambda x: x[1], reverse=True)[:top_n]

    return top_words, top_bigrams


def analyze_sentiments(data):
    reviews = data.review
    sentiments = data.sentiment

    top_bad_sentiment_words, top_bad_sentiment_bigrams = top_words_for_sentiment(reviews, sentiments,
                                                                                 sentiment_type='bad', top_n=10)
    top_good_sentiment_words, top_good_sentiment_bigrams = top_words_for_sentiment(reviews, sentiments,
                                                                                   sentiment_type='good', top_n=10)

    print_top_words(top_bad_sentiment_words, 'Bad Sentiment')
    print_top_words(top_good_sentiment_words, 'Good Sentiment')

    print_top_bigrams(top_bad_sentiment_bigrams, 'Bad Sentiment')
    print_top_bigrams(top_good_sentiment_bigrams, 'Good Sentiment')

    visualize_top_words(top_bad_sentiment_words, 'Bad Sentiment')
    visualize_top_words(top_good_sentiment_words, 'Good Sentiment')



def print_top_words(top_words, title):
    print(f"Top 10 words for {title}:")
    for word_info in top_words:
        if len(word_info) == 2:
            word, count = word_info
            print(f"{word}: {count}")
        else:
            for bigram_info in word_info:
                bigram, count = bigram_info
                print(f"{bigram}: {count}")


def print_top_bigrams(top_bigrams, title):
    print(f"Top 10 bigrams for {title}:")
    for bigram, count in top_bigrams:
        print(f"{bigram}: {count}")


def visualize_top_words(top_words, title):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[word for word, count in top_words], y=[count for word, count in top_words])
    plt.title(f'Top Words for {title}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()



