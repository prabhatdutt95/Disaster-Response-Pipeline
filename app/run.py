
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request

from plotly.graph_objs import Bar
import re
import joblib
from sqlalchemy import create_engine

import json
import plotly


# initializing Flask app
app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    # clean_tokens list: Processed text after normalizing, tokenizing and lemmatizing
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data needed for visuals

    # count messsages based on whether genre is related or not
    genre_related_counts = df[df['related'] == 1].groupby('genre')['message'].count()
    genre_not_related_counts = df[df['related'] == 0].groupby('genre')['message'].count()

    # label for the genre
    genre_names = list(genre_related_counts.index)   

    # Calculate proportion of each category with label = 1
    category_proportion = df[df.columns[4:]].sum()/len(df)
    category_proportion = category_proportion.sort_values(ascending = False)

    # category labels                                                
    category = list(category_proportion.index)
    
    # create visuals
    figures = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_related_counts,
                    name = 'Genre: Related'
                ), 
                Bar(
                    x = genre_names,
                    y = genre_not_related_counts,
                    name = 'Genre: Not Related'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genre and Related Status',
                'xaxis': {
                    'title': "Genre"
                }, 'yaxis': {
                    'title': "Count of Messages"
                }, 'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x = category,
                    y = category_proportion
                )
            ],

            'layout': {
                'title': 'Proportion of Messages <br> by Category',
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45,
                    'automargin': True
                }, 'yaxis': {
                    'title': "Proportion",
                    'automargin': True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, data_set=df)

# web page that handles user query and displays model results
@app.route('/go')

def go():

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    categories = df.columns[4:]
    # print('categories data is',categories)
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(categories, classification_labels))


    positive_results = [key.replace('_', ' ').title() for key,value in classification_results.items() if value == 1]

    # print(positive_results, len(positive_results))
    # print('classification result is',classification_results)

    # This will render the go.html Please see that file. 
    return render_template('go.html',
                            query=query,
                            classification_result=classification_results,
                            positive_results = positive_results
                          )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
