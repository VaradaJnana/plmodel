import pandas as pd
import math
import spacy
nlp = spacy.load('en_core_web_sm')

from topic_modelling.topic_modelling_searching_ensemble import TopicModellingSearchingEnsemble
from topic_modelling.topic_modelling_searching_bertopic import TopicModellingSearchingBERTopic
from topic_modelling.topic_modelling_searching_top2vec import TopicModellingSearchingTop2Vec

class ImprovementExtractor:

    def __init__(self, filepath: str="templates/static/data-files/attribute_negative_counts.csv", num_areas: int=10):
        self.NUM_LINKS = len(pd.read_csv("product_links.csv"))
        self.avoid_words = {'product', 'problem', 'right', 'wrong', 'complaint', 'complain', 'praise', 'bit'}

        data_df = self.load_data(filepath)
        improvement_areas = self.select_improvement_areas(data_df, num_areas)
        self.result = self.get_search_results(improvement_areas=improvement_areas)
        self.export_data(result=self.result)


    def load_data(self, filepath: str):
        """
        Takes the filepath to a CSV file (attribute_negative_counts.csv).
        Returns a pandas DataFrame which contains the data from this filepath.
        """
        return pd.read_csv(filepath)
    
    def tag_pos(self, attribute: str) -> str:
        """
        Takes an attribute (string).
        Returns the Part-of-Speech tag of this word (attribute)
        """
        doc = nlp(attribute)
        return doc[0].pos_
    
    def select_improvement_areas(self, data_df, num_areas: int=10) -> list:
        """
        Takes a pandas DataFrame (data_df).
        Takes the maximum number of improvement areas (num_areas) to be returned.

        Returns a list of the top improvement areas. Returns upto num_areas
        improvement areas (if there are less improvement areas i.e. if there are
        less product features with a net negative sentiment score for at least 20%
        of the products, then a lower number of improvement areas is returned).
        """
        result = []
        counter = 0
        for _, df_row in data_df.iterrows():
            if df_row['score'] >= math.floor(self.NUM_LINKS * 0.20):
                if df_row['attribute'] not in self.avoid_words and self.tag_pos(df_row['attribute']) in {'NOUN', 'VERB'}:
                    counter += 1
                    result.append(df_row['attribute'])
            if counter == num_areas:
                break
        return result
    
    def get_search_results(self, improvement_areas: list, num_reviews: int=10) -> dict:
        """
        Takes 2 inputs:
        - improvement areas: a list of improvement areas
        - num_reviews: the number of custromer reviews to be returned pertaining to each
            of the improvement_areas.
        
        Gets num_reviews customer reviews pertaining to each of the improvement areas in
        improvement_areas. Then, formats these improvement areas and their corresponding
        customer reviews into a dictionary and returns that dictionary (search_results).
        """
        search_results = {'improvementArea': [], 'reviews': []}
        top_n = max(1, num_reviews // 5)
        topic_modelling_filepath = "templates/static/data-files/review_data.csv"
        tmsb = TopicModellingSearchingBERTopic(run_repl=False)
        tmst = TopicModellingSearchingTop2Vec(filepath=topic_modelling_filepath, run_repl=False)
        tmse = TopicModellingSearchingEnsemble(run_repl=False, make_general_fig=False)
        for improvement_area in improvement_areas:
            bert_result = tmsb.doc_search_by_query(topic_modelling_results=tmsb.topic_modelling_results, query=improvement_area, top_n=top_n)
            tv_result = tmst.document_search_by_keywords(model=tmst.model, keywords=improvement_area.split(), unprocessed_documents=tmst.documents, num_docs=num_reviews)
            search_result = tmse.merge_doc_search_result(bert_result=bert_result, tv_result=tv_result, target=num_reviews-top_n)
            search_results['improvementArea'].append(improvement_area)
            search_results['reviews'].append(search_result)
        return search_results
    
    def export_data(self, result: dict) -> None:
        """
        Takes a dictionary (result) with two keys: 'improvementArea' and 'reviews'.
        Each of these keys map to lists with corresponding improvement areas and
        customer reviews.
        Converts the data from the result dictionary into a pandas DataFrame, and
        then exports it as a CSV file: improvement_areas.csv
        """
        pd.DataFrame.from_dict(result).to_csv('templates/static/data-files/improvement_areas.csv')
