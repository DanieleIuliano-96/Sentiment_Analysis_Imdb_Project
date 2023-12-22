# analysis.py
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def top_words_for_sentiment(reviews, sentiments, sentiment_type='bad', top_n=10):
    sentiment_reviews = [review for review, sentiment in zip(reviews, sentiments)
                         if sentiment == 0] if sentiment_type == 'bad' else \
        [review for review, sentiment in zip(reviews, sentiments) if sentiment == 1]

    all_words = ' '.join(sentiment_reviews).split()
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)

    return top_words


def analyze_sentiments(data):
    reviews = data.review
    sentiments = data.sentiment

    top_bad_sentiment_words = top_words_for_sentiment(reviews, sentiments, sentiment_type='bad', top_n=10)
    top_good_sentiment_words = top_words_for_sentiment(reviews, sentiments, sentiment_type='good', top_n=10)

    print_top_words(top_bad_sentiment_words, 'Bad Sentiment')
    print_top_words(top_good_sentiment_words, 'Good Sentiment')

    visualize_top_words(top_bad_sentiment_words, 'Bad Sentiment')
    visualize_top_words(top_good_sentiment_words, 'Good Sentiment')


def print_top_words(top_words, title):
    print(f"Top 10 words for {title}:")
    for word, count in top_words:
        print(f"{word}: {count}")


def visualize_top_words(top_words, title):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=[word for word, count in top_words], y=[count for word, count in top_words])
    plt.title(f'Top Words for {title}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

    sentiment_words = set([word for word, count in top_words])

    plt.figure(figsize=(10, 5))
    plt.bar([word for word in sentiment_words],
            [count for word, count in top_words if word in sentiment_words])
    plt.title(f'Top Words for {title} (Excluding Common Words)')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
