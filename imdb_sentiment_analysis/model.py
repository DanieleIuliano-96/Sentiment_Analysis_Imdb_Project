# model.py
import os

from pandas.io import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


def train_and_evaluate_model(model, vectorizer, X_train, y_train, X_test, y_test):
    vectorizer.fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report


def train_and_save_models(data):
    save_dir = 'model_result'  # Relative path to the project directory

    X = data['review']
    y = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Multinomial Naive Bayes
    nb_model = MultinomialNB()
    count_vectorizer = CountVectorizer(max_features=5000)
    nb_accuracy, nb_report = train_and_evaluate_model(nb_model, count_vectorizer, X_train, y_train, X_test, y_test)
    print("Multinomial Naive Bayes Accuracy:", nb_accuracy)
    print("Classification Report:\n", nb_report)
    save_model(nb_model, count_vectorizer, save_dir, 'nb_model')

    # Support Vector Machine
    svm_model = SVC(kernel='linear', C=1.0)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    svm_accuracy, svm_report = train_and_evaluate_model(svm_model, tfidf_vectorizer, X_train, y_train, X_test, y_test)
    print("\nSupport Vector Machine Accuracy:", svm_accuracy)
    print("Classification Report:\n", svm_report)
    save_model(svm_model, tfidf_vectorizer, save_dir, 'svm_model')

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_accuracy, rf_report = train_and_evaluate_model(rf_model, count_vectorizer, X_train, y_train, X_test, y_test)
    print("\nRandom Forest Classifier Accuracy:", rf_accuracy)
    print("Classification Report:\n", rf_report)
    save_model(rf_model, count_vectorizer, save_dir, 'rf_model')


def save_model(model, vectorizer, save_dir, model_name):
    # Save the model and vectorizer using pickle
    model_filename = os.path.join(save_dir, f'{model_name}_model.pkl')
    vectorizer_filename = os.path.join(save_dir, f'{model_name}_vectorizer.pkl')

    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(vectorizer_filename, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
