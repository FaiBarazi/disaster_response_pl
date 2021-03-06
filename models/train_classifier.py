import sys
import string

from joblib import dump

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Loads the dataset from an sqlite db and splits it into
    features and labels.

    Args:
        database_filepath(str): Path to sqlite file.

    Returns:
        X (Dataframe): Dataframe of features, messages in this case.
        y (Dataframe): Dataframe of labels one-hot encoded.
        category_names(list): list of all the categories/columns in y.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response', con=engine)
    X = df['message']
    y = df.drop(columns=['message', 'original', 'id', 'genre'])
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    """
    Tokenizes and lemmatizes the text. Tokenization splits the text into
    separate words. Lemmatization converts the words into their dicitonary
    form.

    Args:
        text(str): The text to tokenize.

    Returns:
        cleaned_tokens(List): text tokenized and lemmatized into a list of
        words.
    """
    tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text.lower()))
    lammetizer = WordNetLemmatizer()
    cleaned_tokens = [
        lammetizer.lemmatize(tok).lower().strip().strip(string.punctuation) for tok in tokens
    ]
    return cleaned_tokens


def build_model():
    """
    Runs a model pipeline with a multioutput classifier
    and does a gridsearch to optimize the hyperparameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
        'vect__max_df': (0.5, 0.75),
        'tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    calculates and displays the f-score precision and recall of
    the model.

    Args:
        model(scikitlearn model): multiouput trained model .
        X_test(Dataframe): Pandas dataframe of test features.
        Y_test(DataFrame): pandas dataframe of test labels.
        category_names(list): list of category names to predict.

    Returns:
        Void

    """
    y_hat = model.predict(X_test)
    print(classification_report(Y_test.values, y_hat, target_names=category_names))


def save_model(model, model_filepath):
    """
    Use scikit learn joblib to save the model with its parameters.
    This is more effiecent than normal python pickling as per the scikit
    docs here: https://scikit-learn.org/stable/modules/model_persistence.html.

    Args:
        model(estimator): Scikit learn estimator trained model.
        model_filepath(string/path-like object): A path to where the model is going
            to be saved. file extention is `.joblib`.

    Returns:
        void, dumps a joblib pickled file based on the model_filepath.

    """
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {(model_filepath)}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
