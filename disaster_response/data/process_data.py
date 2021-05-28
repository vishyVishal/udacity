import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Inputs:
        messages_filespath - path to csv file for messages
        categories_filespath - path to csv file for categories
       
    Output:
        df - Pandas Dataframe of the merged dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    """
    Input:
       df - dataframe of the the merged dataset
       
    Output:
        df - Pandas Dataframe of the merged dataset after some cleaning/preprocessing
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories[:1]
    
    # use this row to extract a list of new column names for categories.
    
    category_colnames = row.values[0]
    category_colnames = [name[:-2] for name in category_colnames]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #  Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')
        
    # Replace categories column in df with new category columns
    # drop the original categories column from `df`

    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Removes rows where Related==2
    df = df[df['related'] != 2]
    
    return df

def save_data(df, database_filename):
    """
    Save the dataframe to a SQL Database
    Inputs:
        df - cleaned dataframe
        database_filename - name of SQL Database
       
    Output:
        df - Pandas Dataframe of the merged dataset
    """
    engine = create_engine('sqlite:///' + database_filename)
    
    df.to_sql('CorrectedTable', engine, index=False)

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