import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load the data from csv files and return a DataFrame
    of the two  merged on id.
    Args:
        messages_filepath(str): csv file of messages
        categories_filepaht(str): csv file of categories

    return:
        void
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Gets a DataFrame from the load_data output
    and cleans it furhter.
    Args:
        df(Dataframe): Pandas Dataframe

    Returns:
        df after being transformed and cleaned.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # the values are in the form `value-0`, here we get rid of the
    # last 2 values.
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # replace anything greater than 1 in 'related column' by 1
    categories['related'] = categories['related'].apply(lambda x: 1 if x > 1 else x)
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    df.drop_duplicates(inplace=True)

    df = pd.concat([df, categories], axis=1)
    df.dropna(subset=['message'], inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves the data frame to an sqlite dataframe.

    Args:
        df(Dataframe): Pandas dataframe.
        database_filename(str): path to sqlite path.

    Returns:
        void
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
