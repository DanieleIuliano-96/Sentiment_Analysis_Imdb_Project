# Sentiment Analysis Web App

This project implements a simple web application for sentiment analysis using various machine learning models. Users can input a review, choose a model, and get the sentiment prediction.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following prerequisites installed on your machine:

- Python 3.x
- Flask
- scikit-learn
- Other dependencies (if any)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://DanieleIuliano-96/Sentiment_Analysis_Imdb_Project.git
    cd sentiment-analysis-web-app
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask application:**

    ```bash
    python app.py
    ```

4. **Before running the application, unzip the IMDB-Dataset.zip file in the same directory. This step is necessary to have the dataset available for sentiment analysis.**

5. **Optionally, you can create the models using `create_models.py`. Run the following command to create models:**

    ```bash
    python create_models.py
    ```

    **Alternatively, if you don't want to create the models, you can unzip the file `models_list_of_models.zip` to use the pre-trained models directly in `app.py`.**

6. **Open your web browser and navigate to `http://localhost:5000`.**

## Models

This project includes the following sentiment analysis models:

- Naive Bayes
- Support Vector Machine
- Random Forest
