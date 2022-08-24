import os
import pandas as pd

from topic_modelling.train_bertopic_model import TrainBERTopicModel
from topic_modelling.topic_modelling_results import TopicModellingResults


class TopicModellingSearchingBERTopic:

    def __init__(self, filepath: str="", run_repl: bool=True, model_already_trained: bool=True, generate_general_fig: bool=False):
        model_filepath = f"{os.getcwd()}/topic_modelling/bertopic_model"
        if not model_already_trained:
            data_df = self.load_data(filepath)
            TrainBERTopicModel(data_df, output_filepath=model_filepath)

        self.topic_modelling_results = TopicModellingResults(model_filepath)
        if generate_general_fig:
            self.generate_topic_overview_figure(self.topic_modelling_results)

        print("Completed bertopic")

    def load_data(self, filepath: str):
        """
        Takes the filepath to a CSV file. This should be the path to the file with the
        webscraped customer reviews.
        Reads the data from the file into a pandas DataFrame, and fills in any N/A
        values in the reviewHeader or reviewText columns with an empty string.
        Returns the resulting DataFrame.
        """
        df = pd.read_csv(filepath)
        df[['reviewHeader', 'reviewText']] = df[['reviewHeader', 'reviewText']].fillna("")
        return df
    
    def generate_topic_overview_figure(self, topic_modelling_results) -> None:
        """
        Takes an object of the TopicModellingResults class.
        Uses this object to generate an image of the top words associated with 
        each of the topics created by BERTopic model.
        (Note: this image is for the potential reference of developers; it is not
        displayed on the dashboard webapp).
        """
        topic_modelling_results.generate_topic_space_overview(topic_modelling_results.model)
    
    def generate_general_topic_wordcloud_by_query(self, topic_modelling_results, query: str, top_n: int=5) -> None:
        """
        Takes 3 inputs:
        - topic_modelling_results: an object of the TopicModellingResults class.
        - query: a string, the search query entered by the user
        - top_n: the number of topics or customer reviews from which we will get
                the words for the wordclouds
        Generates 2 wordclouds:
        1. a wordcloud of the top words present in the customer reviews from the top_n
            topics that are most relevant to the search query
        2. a wordcloud of all the words present in the top_n customer reviews that
            are found to be most relevant to the search query
        """
        topic_modelling_results.generate_wordcloud_for_query(topic_modelling_results.model, query, top_n)
        topic_modelling_results.generate_second_wordcloud_for_query(topic_modelling_results.model, query, top_n)
    
    def doc_search_by_query(self, topic_modelling_results, query: str, top_n: int=5) -> list:
        """
        Takes 3 inputs:
        - topic_modelling_results: an object of the TopicModellingResults class.
        - query: a string, the search query entered by the user
        - top_n: the number of topics from which we will get the words for the wordclouds
        Returns a list of the most relevant customer reviews from each of the top_n
        topics that are most relevant to the query.
        """
        return topic_modelling_results.document_search_by_query(topic_modelling_results.model, query, top_n)
    