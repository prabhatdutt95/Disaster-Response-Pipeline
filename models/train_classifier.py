# import libraries
import sys
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from random import randrange

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    
    """
    Loads data from SQL Database

    Args:
    database_filepath: SQL database file

    Returns:
    X (pandas_dataframe): Features dataframe
    Y (pandas_dataframe): Target dataframe
    category_names (list): Target labels 
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con = engine)

    # Defining feature and target dataframes
    X,Y = df['message'], df.iloc[:,4:]

    # Target labels
    category_names = Y.columns

    # Mapping any values except 0/1 for each column with randomly selected 0/1
    for column in category_names:
        Y[column] = Y[column].map(lambda x: randrange(0,2) if x > 1 or x < 0 else x)
    
    return X, Y, category_names 

def tokenize(text):

    """
    Tokenizes text data

    Args:
    text (str): Messages as text data

    Returns:
    words (list): Processed text after normalizing, tokenizing and lemmatizing
    """

    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    """
    Build model with GridSearchCV
    
    Returns:
    Trained model after performing grid search
    """
    
    # model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer()), ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])

    # parameters added
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0),
                  'vect__max_features': (None, 5000, 10000),
                  'tfidf__use_idf': (True, False),
                  'clf__estimator__estimator__tol': (0.0001, 0.0002),
                }

    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate model's perfomance on test data and generate report by using classification_report

    Args:
    model: Trained ML model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels

    Returns:
    none - displays scores(precision, recall, f1-score) for each output category of the dataset
    """

    # Predict the target value for test feature
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))    

    # print accuracy score and best model parameter
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    
    """
    This function saves the model to a Python pickle file

    Args:
    model: Trained model
    model_filepath: Location to save the model

    Returns:
    none - Saves the model to pickle file
    """

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()