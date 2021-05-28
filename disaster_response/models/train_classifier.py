import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, make_scorer

import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.base import TransformerMixin


# Class Needed for Model
class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()



def load_data(database_filepath):
    """
    Input:
        database_filespath - path to csv file for messages
        
       
    Outputs:
        X - Train Dataframe 
        Y - Label Dataframe
        category_names -> output categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Load the data from the database
    df = pd.read_sql_table('CorrectedTable', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """Preprocess and clean text string
    
    Input:
    text - String containing message for processing
       
    Output:
    tokens- List containing normalized and lemmatize word tokens
    """
    # Detect URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize and tokenize and remove punctuation
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


def build_model():
    """
    Input:
        None        
       
    Output:
       Model Pipeline
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('to_dense', DenseTransformer()), 
                ('clf', MultiOutputClassifier( HistGradientBoostingClassifier(max_iter=100))
                )
            ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Inputs:
        model- Classifier
        X_test- Messages
        Y_test- Disaster categories labels
        category_names- Disaster category names.
    """
    
    y_pred = model.predict(X_test)
    Y = Y_test
    
    acc_lst = []
    f1_lst = []

    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        accuracy = accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])
        print('Accuracy {}\n\n'.format(accuracy))
        acc_lst.append((Y.columns[i], accuracy))
        print('F1 {}\n\n'.format(f1_score(Y_test.iloc[:, i].values, y_pred[:, i],average='weighted')))
        f1_lst.append((Y.columns[i],f1_score(Y_test.iloc[:, i].values, y_pred[:, i],average='weighted')))
        
    # Print Some metrics     
    acc_scores = [ item[1] for item in acc_lst]
    print("Accuracy Overall: " + str( sum(acc_scores)/len(acc_scores) ))
    
    f1_scores = [ item[1] for item in f1_lst]
    print("F1 Score Overall: " + str( sum(f1_scores)/len(f1_scores)) )
    


def save_model(model, model_filepath):
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