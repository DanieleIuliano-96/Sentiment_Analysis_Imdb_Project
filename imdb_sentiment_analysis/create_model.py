# create_model.py
import preprocessing
import model
import analysis

if __name__ == "__main__":
    # Load data
    data = preprocessing.read_data('resources/IMDB-Dataset.csv')

    # Preprocess data
    data = preprocessing.preprocess_data(data)

    # Train and Save model
 #   model.train_and_save_models(data)

    # Analyze sentiments
    analysis.analyze_sentiments(data)
