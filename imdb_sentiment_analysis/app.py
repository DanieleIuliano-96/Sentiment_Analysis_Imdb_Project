# app.py
import pickle

from flask import Flask, request, render_template

from preprocessing import clean, is_special, to_lower, rem_stopwords, stem_txt

app = Flask(__name__)


# Load models and vectorizers using pickle
def load_model(model_name):
    model_filename = f'model_result/{model_name}_model.pkl'
    vectorizer_filename = f'model_result/{model_name}_vectorizer.pkl'

    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_filename, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


# Model endpoints
def predict_sentiment(model_name, review):
    cleaned_review = stem_txt(rem_stopwords(to_lower(is_special(clean(review)))))
    model, vectorizer = load_model(model_name)
    review_vectorized = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vectorized)[0]
    return prediction


@app.route('/')
def index():
    return render_template('index.html', sentiment=None)


@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    model_name = request.form['model']
    sentiment = predict_sentiment(model_name, review)
    return render_template('index.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
