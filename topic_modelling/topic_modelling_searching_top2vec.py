import pandas as pd

from top2vec import Top2Vec

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

import string

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import os


class TopicModellingSearchingTop2Vec:

    def __init__(self, filepath: str, speed: str='deep-learn', models_already_trained: bool=True, run_repl: bool=True):
        # os.environ["TFHUB_CACHE_DIR"] = "/var/folders/cn/dtb98nld0j7g5gfysyrv8y3m0000gp/T/tfhub_modules/063d866c06683311b44b4992fd46003be952409c"
        self.data_df = self.load_data(filepath)
        self.documents = self.generate_document_list(self.data_df)

        self.stop_words = set(stopwords.words('english'))
        processed_documents = self.preprocess_data(self.documents)

        if not models_already_trained:
            model = self.generate_model(processed_documents, speed)
            model.save("topic_modelling/main_model")
            print(f"Number of topics: {model.get_num_topics()}")
        else:
            self.model = Top2Vec.load('topic_modelling/main_model')
        
        print("Completed top2vec")


    def load_data(self, filepath: str):
        """
        Takes the filepath to a CSV file. This should be the path to the file with the
        webscraped customer reviews.
        Reads the data from the file into a pandas DataFrame, and fills in any N/A
        values in the reviewHeader or reviewText columns with an empty string.
        Returns the resulting DataFrame.
        """
        df =  pd.read_csv(filepath)
        df[['reviewHeader', 'reviewText']] = df[['reviewHeader', 'reviewText']].fillna('')
        return df

    
    def generate_document_list(self, data_df) -> list:
        """
        Takes a pandas DataFrame (data_df) with the data of the webscraped customer
        reviews.
        Creates a list (documents), which contains the text from each customer
        review as a separate list item. Each list item contains the text from
        the reviewHeader and the reviewText columns of the corresponding row of
        data_df. If the reviewHeader column does not exist, then only the text from
        the reviewText column is returned.
        This list (documents) is then returned.
        """
        documents = []
        for _, row in data_df.iterrows():
            try:
                documents.append(str(row['reviewHeader']) + '. ' + str(row['reviewText']))
            except:
                documents.append(str(row['reviewText']))
        return documents
    

    def remove_stopwords_and_punctuation(self, review: str) -> str:
        """
        Takes a string (review).
        Returns the string after removing punctuation, and removing all
        of the standard nltk stopwords and then converting the string
        to lowercase.
        """
        return " ".join([word.lower() for word in word_tokenize(review) \
            if (word.lower() not in self.stop_words and word.lower() not in string.punctuation)])
    
    def preprocess_data(self, documents: list) -> list:
        """
        Takes a list of strings (documents).
        Returns a list of strings after processing each string in the list.
        Processing involves stopwords removal, punctuation removal, and
        conversion to lowercase.
        """
        return list(map(self.remove_stopwords_and_punctuation, documents))
    

    def generate_model(self, documents: list, speed: str, min_count: int=10):
        """
        Takes 3 inputs:
        - documents: a list of strings (documents), which is the preprocessed
            list of customer reviews
        - speed: a string, depicting the speed at which the top2vec model is
            going to be trained
        - min_count: an int, a frequency with which words with a lower frequency
            than min_count are ignored.
        """
        if speed not in {'fast-learn', 'learn', 'deep-learn'}:
            raise ValueError(f"Value passed for 'speed' parameter is invalid.\nEntered value must be one of {{'fast-learn', 'learn', 'deep-learn'}}")
        return Top2Vec(documents=documents, speed=speed, workers=8, min_count=min_count, embedding_model='universal-sentence-encoder')
    

    def document_search_by_keywords(self, model, keywords: list, unprocessed_documents: list, num_docs: int=5) -> list:
        """
        Takes 4 inputs:
        - model: the Top2Vec model
        - keywords: a list of strings, where each string is a word from the 
            search query
        - unprocessed_documents: a list of strings, where each string is a customer
            review that has not yet been preprocessed.
        - num_docs: an int, the total number of customer reviews to be returned

        Returns a list of the top customer reviews that are most closely related
        to the search query (from keywords)
        """
        result = []
        try:
            docs, document_scores, document_ids = model.search_documents_by_keywords(keywords=keywords, num_docs=num_docs)
            for _, score, doc_id in zip(docs, document_scores, document_ids):
                result.append((score, unprocessed_documents[doc_id]))
            return result
        except:
            return result
