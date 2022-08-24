import pandas as pd
from ast import literal_eval

from product_ranking.product_ranker import ProductRanker
from topic_modelling.topic_modelling_searching_ensemble import TopicModellingSearchingEnsemble
from topic_modelling.topic_modelling_searching_ensemble import TopicModellingSearchingBERTopic
from topic_modelling.topic_modelling_searching_ensemble import TopicModellingSearchingTop2Vec

class DataFetcher:
    def __init__(self):
        pass
    
    # Getting the Top 20 Attributes List
    def get_attribute_word_list(self, data_df) -> list:
        """
        Takes a pandas DataFrame (data_df) containing the data from the file
        top_twenty_attributes.csv
        Returns a list of tuples where each tuple contains one of the top
        20 product features (attributes) that customers care about, and the relevance
        score of that product feature.
        (Note: the list is sorted in decreasing order of relevance scores)
        """
        data_list = []
        for _, df_row in data_df.iterrows():
            data_list.append((df_row['Word'], round(float(df_row['TfIdfScore']))))
        return data_list

    def get_top_twenty_attributes(self) -> list:
        """
        Returns a list of tuples of the top 20 attributes (product features)
        that customers care about, and their corresponding relevance
        scores.
        Fetches this data from top_twenty_attributes.csv
        """
        filepath = "templates/static/data-files/top_twenty_attributes.csv"
        data_df = pd.read_csv(filepath)
        return self.get_attribute_word_list(data_df)
    

    # Getting the top attributes listed by Amazon
    def get_amazon_suggested_attributes(self) -> list:
        """
        Returns a list of the attributes (product features) suggested by Amazon
        as being potentially relevant to customers.
        Fetches this data from amazon_suggested_attributes.csv
        """
        filepath = "templates/static/data-files/amazon_suggested_attributes.csv"
        data_df = pd.read_csv(filepath)
        return data_df['attribute'].to_list()
    

    # Getting Products List
    def get_products_list(self):
        """
        Returns a list of all of the products mentioned in
        product_attribute_descriptions_report.csv
        Each of the items in the list is a tuple, where the first value in the tuple
        is the product id, and the second value in the tuple is the product name
        (Note: the product names are truncated for better visual design on the
        dashboard webapp)
        """
        filepath = "templates/static/data-files/product_attribute_descriptions_report.csv"
        data_df = pd.read_csv(filepath)
        products_list = []
        for _, df_row in data_df.iterrows():
            products_list.append((int(df_row['productID']), self.get_product_name(df_row['productName'])))
        return products_list


    # Getting Product-Attribute Descriptions data
    def get_df_row(self, data_df, product_id):
        """
        Inputs:
        - data_df: a pandas DataFrame, which holds the information that was contained in
                product_attribute_descriptions_report.csv
        - product_id: the product id of the porduct currenlty being considered

        Returns the row of data_df that corresponds to the entered product id.
        If no such row can be found, then a ValueError is raised.
        """
        for _, df_row in data_df.iterrows():
            if int(df_row['productID']) == product_id:
                return df_row
        raise ValueError("An incorrect product_id was entered", product_id)
    
    def get_product_name(self, product_name: str, char_limit: int=50) -> str:
        """
        Takes the name of a product. Also takes a character limit (char_limit).
        
        If the length of the product name is less than the character limit, then
        returns the name as is. If the name is longer than the character limit,
        then takes the first char_limit characters of the name, adds an ellipses
        to the end of this truncated name, and returns the resulting string.
        """
        if len(product_name) < char_limit:
            return product_name
        return product_name[:char_limit] + " ..."
    
    def get_top_features(self, top_features_list):
        """
        Takes a list of the top product features that customers care about
        (top_features_list).
        Returns a string in which all of these features are listed and
        separated by a comma. Additionally, any underscores in the product
        feature names are replaced with whitespaces.

        The resulting string with comma-separated product features is
        returned.
        """
        return ", ".join(feature.replace("_", " ") for feature in top_features_list)
    
    def format_feature_descriptions(self, feature_descriptions: dict) -> tuple:
        """
        Takes a dictionary (feature_descriptions) which maps product features
        to the descriptive words used to describe a certain product with
        respect to that feature.
        Currently, each product feature is mapped to a list of descriptive words.
        This function takes this list of descriptive words and replaces it with a
        string, which contains each of these descriptive words, separated by a 
        comma.
        The function also maintains a record of which features had no descriptive
        words corresponding to them (no_descriptions).
        The function finally returns a tuple, where the first value is the
        modified feature_descriptions dictionary, and the second value is the list
        no_descriptions.
        """
        no_description = []
        for entry in feature_descriptions:
            feature_descriptions[entry] = ", ".join(description.replace("_", " ") for description in feature_descriptions[entry])
            if feature_descriptions[entry].strip() == "":
                no_description.append(entry)
        for entry in no_description:
            feature_descriptions.pop(entry)
        return feature_descriptions, no_description
    
    def round_values(self, attribute_scores: dict, decimal_places: int=2) -> dict:
        """
        Inputs:
        - attribute_scores: a dictionary which has attributes (product features)
            as keys, and the corresponding sentiment scores as values.
        - decimal_places: an int, representing how many decimal places we want
            to round off our values to
        
        This function takes the sentiment score that each attribute maps to,
        and rounds it off to the specified number of decimal places.
        If the value is greater than 1, then it is reduced to 1.00
        If the value is less than -1, then it is increased to -1.00

        The dictionary with these rounded off values is returned.
        """
        for attribute in attribute_scores:
            attribute_scores[attribute] = max(-1.00, min(1.00, float(round(attribute_scores[attribute], decimal_places))))
        return attribute_scores
    
    def include_shift_values(self, feature_scores) -> dict:
        """
        Takes a dictionary (feature_scores) mapping product features
        against the corresponding sentiment scores.
        For each sentiment score, calculates the shift value that needs
        to be applied to the black triangle pointer on the frontend so
        that it points to the correct part of the colorbar (for the score
        visualization).
        Currently, each feature in feature_scores is mapped to its sentiment
        score. Replaces this sentiment score with a tuple, whose first value
        is the same sentiment score, and whose second value is the shift string.

        Returns this new (modified) dictionary (feature_scores)
        """
        for feature in feature_scores:
            score = feature_scores[feature]
            if score == 0.0:
                score = abs(score)
                shift = "left: 0.75%;"
            elif score > 0.0:
                shift = "left: " + str((33 * score)) + "%;"
            else:
                shift = "right: " + str(abs(33 * score)) + "%;"
            feature_scores[feature] = (feature_scores[feature], shift)
        return feature_scores
    
    def get_product_info(self, df_row) -> dict:
        """
        Takes a single row (df_row) of the pandas DataFrame which contains data from
        product_attribute_descriptions_report.csv
        This single row contains the information corresponding to a single product.
        Returns a dictionary which contains information on:
        - the product name
        - the top product features (for this product)
        - the top descriptions for each fo the product features (for this product)
        - the sentiment scores for each of the product features (for this product)
        Note: The function filters out information related to any features that have
            no corresponding desriptive words
        """
        product_info = dict()
        product_info['productName'] = self.get_product_name(df_row['productName'], 70)
        product_info['topFeatures'] = literal_eval(df_row['topRelevantAttributes'])
        product_info['featureDescriptions'], no_description = self.format_feature_descriptions(literal_eval(df_row['topAttributeDescriptions']))
        product_info['featureScores'] = self.include_shift_values(self.round_values(literal_eval(df_row['attributeScores'])))
        for feature in no_description:
            if feature in product_info['featureScores']:
                product_info['featureScores'].pop(feature)
            if feature in product_info['topFeatures']:
                product_info['topFeatures'].remove(feature)
            if feature in product_info['featureScores']:
                product_info['featureScores'].pop(feature)
        product_info['topFeatures'] = self.get_top_features(product_info['topFeatures'])
        return product_info

    def get_product_attribute_description_data(self, product_id) -> dict:
        """
        Takes a product id (product_id).
        Returns the product attribute description data for the product
        corresponding to that product id. This data is in the form of 
        a dictionary, and it includes information on:
        - the product name
        - the top product features (for this product)
        - the top descriptions for each fo the product features (for this product)
        - the sentiment scores for each of the product features (for this product)
        This dictionary is populated with data from product_attribute_descriptions_report.csv
        """
        filepath = "templates/static/data-files/product_attribute_descriptions_report.csv"
        data_df = pd.read_csv(filepath)
        df_row = self.get_df_row(data_df, product_id)
        return self.get_product_info(df_row)
    

    # Get product ranker info
    def get_features_set(self) -> set:
        """
        Gets a set of all of the unique product features that are mentioned as
        being amongst the top 10 product features for any of the products in
        product_attribute_descriptions_report.csv
        Returns this set of product features.
        """
        filepath = "templates/static/data-files/product_attribute_descriptions_report.csv"
        data_df = pd.read_csv(filepath)
        features_set = set()
        for _, df_row in data_df.iterrows():
            for feature in literal_eval(df_row['topRelevantAttributes']):
                features_set.add(feature.replace("_", " "))
        return features_set
    
    def get_product_links_dict(self) -> dict:
        """
        Prepares a dictionary which maps eahc product id against the Amazon link
        of the product corresponding to that product id.
        Returns this dictionary (product_links)
        """
        filepath = "product_links.csv"
        data_df = pd.read_csv(filepath)
        product_links = dict() # Dict: product_id -> product_link
        for idx, df_row in data_df.iterrows():
            product_links[idx + 1] = df_row['productLinks']
        return product_links

    def get_product_rank_info(self, attribute: str) -> list:
        """
        Takes an attribute (product feature), which is a string.
        Returns a list of the products for which the given attribute
        was deemed to be one of the top 10 most relevant product features.

        The values in the list are sorted in decreasing order of sentiment
        scores of the product with respect to the given attribute, so it
        is a ranklist of how much customers like the products with respect
        to the given attribute.

        Each item in the list is a tuple, which consists of:
        - the product id
        - the product name
        - the descriptive words used for the product with respect to the
            given attribute
        - the sentiment score of the product for the given attribute
        - the shift value, for the triangle pointer in the visualization
            with the colorbars (so the pointer points to the correct part
            of the colorbar)
        - the link to the Amazon page for the product
        """
        prod_rank_df = ProductRanker(attribute).result
        product_links_dict = self.get_product_links_dict()
        product_ranker_info = []
        for _, df_row in prod_rank_df.iterrows():
            product_id = int(df_row['productID'])
            print("id", product_id)
            product_name = self.get_product_name(df_row['productName'], 70)
            descriptive_words = ", ".join(word.replace("_", " ") for word in df_row['descriptiveWords'])
            if descriptive_words.strip() == "":
                continue
            score = max(-1.00, min(1.00, round(float(df_row['score']), 2)))
            if score == 0.0:
                score = abs(score)
                shift = "left: 0.75%;"
            elif score > 0.0:
                shift = "left: " + str((33 * score)) + "%;"
            else:
                shift = "right: " + str(abs(33 * score)) + "%;"
            link = product_links_dict[product_id]
            product_ranker_info.append((product_id, product_name, descriptive_words, score, shift, link))
        return product_ranker_info
    
    

    # Get market improvement areas info
    def filter_relevant_reviews(self, must_contain: str, reviews) -> list:
        """
        Inputs:
        - must_contain: a string
        - reviews: a list of customer reviews

        Returns a list of all the customer review in reviews that contain the
        string must_contain.
        (Note: this filter is applied to ensure that customer reviews shown
        actually talk about the improvement area)
        """
        return [review for review in reviews if must_contain in review.lower()]

    def get_market_improvement_areas_info(self) -> list:
        """
        Returns a list of tuples, where the first item in each tuple is an
        improvement area, and the second value in each tuple is a list of
        customer reviews pertaining to that improvement area
        """
        filepath = "templates/static/data-files/improvement_areas.csv"
        data_df = pd.read_csv(filepath)
        improvement_areas_info = []
        for _, df_row in data_df.iterrows():
            improvement_area = df_row['improvementArea']
            reviews = self.filter_relevant_reviews(improvement_area.lower(), literal_eval(df_row['reviews']))
            improvement_areas_info.append((improvement_area, reviews))
        return improvement_areas_info
    

    # Get customer review searching data
    def is_some_contained(self, query: str, result: str) -> bool:
        """
        Takes a search query (string).
        Also takes a customer review (result).
        Returns True if any of the terms from the search query are
        contained within the customer review (result).
        Else, returns False
        """
        query_terms = query.split()
        for term in query_terms:
            if term in result:
                return True
        return False

    def get_customer_review_search_results(self, query: str, num_reviews: int=10, \
            filepath: str="templates/static/data-files/review_data.csv", speed: str="deep-learn") -> list:
        """
        Inputs:
        - query (str): the search query
        - num_reviews (int): the maximum number of customer reviews to be returned
        - filepath (str): the path to the file with the scraped customer reviews
        - speed (str): specifying the speed with which to run the Top2Vec model

        Gets the most relevant customer reviews to the search query (the number of
        customer reviews is num_reviews). Then, filters out reviews in which none
        of the terms from the search query are mentioned, and then returns a list
        of these customer reviews
        """
        try:
            tmse = TopicModellingSearchingEnsemble(run_repl=False, make_general_fig=False)
            top_n = max(1, num_reviews // 5)
            tmsb = TopicModellingSearchingBERTopic(run_repl=False)
            bert_result = tmsb.doc_search_by_query(topic_modelling_results=tmsb.topic_modelling_results, query=query, top_n=top_n)
            tmst = TopicModellingSearchingTop2Vec(filepath=filepath, speed=speed, run_repl=False)
            tv_result = tmst.document_search_by_keywords(model=tmst.model, keywords=query.split(), unprocessed_documents=tmst.documents, num_docs=num_reviews)
            result = tmse.merge_doc_search_result(bert_result, tv_result, num_reviews-top_n)
            result = [review for review in result if self.is_some_contained(query.lower(), review.lower())]
        except KeyError:
            return ["An error occurred! This is probably because there is not enough data to properly run search-systems"]
        return result

    
    # Get wordcloud search data
    def get_wordcloud_search_results(self, query: str, num_topics: int=3):
        """
        Inputs:
        - query (str): the search query
        - num_topics (int): the number of topics/customer reviews from which
            the words for the wordcloud will be sourced
        
        Generates two wordclouds containing words relevant to the search query.
        The first contains the top words from the top topics that are most
        relevant to the search query.
        The second contains the words from the top customer reviews pertaining
        to the search query.

        Returns a list of filepaths to these generated wordclouds.
        If an error occurs in this process, it is because there have not been enough
        customer reviews scraped to effectively run topic modelling, and a string
        explaining this error is returned
        """
        try:
            tmsb = TopicModellingSearchingBERTopic(run_repl=False)
            tmsb.generate_general_topic_wordcloud_by_query(tmsb.topic_modelling_results, query, num_topics)
            filepaths = []
            file_prefix = "../static/wordclouds/"
            for i in range(1, 3):
                filepaths.append(file_prefix + f"general_wordcloud_{i}_{'_'.join(word for word in query.split())}.png")
        except KeyError:
            filepaths = "An error occurred! This is probably because there is not enough customer review data!"
        return filepaths
