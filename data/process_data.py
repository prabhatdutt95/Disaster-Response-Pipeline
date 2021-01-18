# import libraries
import sys
import pandas as pd
from random import randrange

from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):

    """
    This method has 2 filepath as input for csv files and merges them into single dataframe
    
    Args:
    messages_file_path (str): Messages CSV filepath
    categories_file_path (str): Categories CSV filepath

    Returns:
    df (pandas_dataframe): Dataframe obtained from merging the two input dataframe on 'id' column
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    """Merge the messages and categories datasets using the common id"""
    df =  messages.merge(categories, on="id")
    
    return df

def clean_data(df):
    
    """
    This method cleans the output dataset 
    
    Args:
    df (pandas_dataframe): Merged dataframe obtained from load_data() function

    Returns:
    df (pandas_dataframe): Cleaned data to be processed further
    """

    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';').tolist())
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0] for x in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    """Convert category values to just numbers 0 or 1"""
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x.split('-')[1] for x in categories[column]]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # convert values to binary to handle multiclass values
        # So if we get 2, we handle this by either assuming 2 as 1
        categories[column] = categories[column].map(lambda x: randrange(0,2) if x > 1 or x < 0 else x)
    
    # Adding id column to categories dataset
    categories['id'] = df['id']
    
    """Replace categories column in df with new category columns"""
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, how = 'outer', on=['id'])
    
    # check number of duplicates
    if df.duplicated().sum() > 0:
        # drop duplicates
        df = df.drop_duplicates(keep="last")
    
    return df

def save_data(df, database_filename):

    """
    Saves the cleaned data to an SQL database

    Args:
    df (pandas_dataframe): Cleaned data returned from clean_data() function
    database_file_name (str): File path of SQL Database into which the cleaned data is to be saved

    Returns:
    None
    """

    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    table_name = database_filename.split('.')[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
