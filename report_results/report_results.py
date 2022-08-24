import pandas as pd
from ast import literal_eval

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

from report_results.synonym_mapper import SynonymMapper
from report_results.attribute_negative_count_mapper import AttributeNegativeCountMapper


class ReportResults:

    def __init__(self, filepath: str="templates/static/data-files/mined_data.csv", product_name: str=""):
        self.filepath = filepath
        self.product_words = self.get_product_synonyms(product_name.split())

        self.data = None
        self.attribute_set = set()
        self.attribute_text_distribution = []
        self.word_count = None
        nlp.Defaults.stop_words |= {"thing", "things", "got", "gone", "going", "took", "love", "like", "luck", "hate", "go", "good", "bad", "right", "wrong", "thank"}
        self.stopwords = nlp.Defaults.stop_words

        self.load_data()
        self.generate_report_data()
        self.get_rating_by_feature_attributes_result()

    def get_product_synonyms(self, product_name_words: list) -> set:
        """
        Takes a list of words (product_name_words).
        Returns a set containing the synonyms of all the words in
        product_name_words.
        Returns this set.
        """
        avoid_words = set()
        for word in product_name_words:
            avoid_words.add(str(word).lower().strip())
            for syn in wordnet.synsets(word):
                for name in syn.lemma_names():
                    avoid_words.add(str(name).lower().strip())
        return avoid_words

    def load_data(self) -> None:
        """
        Loads the data contained in the CSV file at the location contained
        in self.filepath.
        Filters and retains only those rows of data that correspond to
        verified purchases on Amazon (if this column of data exists).
        """
        self.data = pd.read_csv(self.filepath)
        try:
            self.data = self.data.loc[self.data['verifiedPurchase'] == True]
        except:
            pass
    
    def get_attribute_set(self, data, full_corpus: bool=False) -> tuple:
        """
        Takes 2 inputs:
        - data: a pandas DataFrame
        - full_corpus: a boolean
        When full_corpus is True, then data contains the data from self.data (i.e.
        the data from mined_data.csv)

        This function generates a set of all the unique attributes mentioned in data.
        These attributes are mapped against their synonyms and stemmed words so
        occurrances of these synonyms/stemmed versions are replaced with the original
        attribute (to reduce vocabulary size and also reduce the likelihood of having
        multiple similar attributes). This set is called attribute_set.
        The function also generates a list which records the distribution of which
        attributes are mentioned in which review. This is stored as 
        attribute_text_distribution.
        The function also generates a dictionary which records the total Amazon
        helpful count for customer reviews in which an attribute is mentioned.
        This is stored in attribute_helpful_counts.
        Lasltly, the function also generates a list which records the distribution
        of which descriptive words are mentioned in which customer review. This is
        stored as description_text_distribution.

        All four of these generates values are returned in a tuple (in this order)
        """
        sm = SynonymMapper()
        if full_corpus:
            ancm = AttributeNegativeCountMapper()

        attribute_set = set()
        attribute_helpful_counts = dict()
        attribute_text_distribution = []
        description_text_distribution = []
        for _, row in data.iterrows():
            current_attribs_distrib = ""
            current_descriptions_distrib = ""
            mined_text = literal_eval(row['minedText'])
            helpful_count = int(row['reviewHelpfulCount'])
            for entry in mined_text:
                for word in entry['aspect'].split():
                    if word not in self.stopwords and word not in self.product_words and float(entry['polarity']) != 0.0:
                        if full_corpus:
                            ancm.map_entry(word=word, entry=entry, row=row)
                        word = sm.map_synonyms(word=word)
                        attribute_set.add(word)
                        current_attribs_distrib += (word + ' ')
                        if word in attribute_helpful_counts:
                            attribute_helpful_counts[word] += helpful_count
                        else:
                            attribute_helpful_counts[word] = helpful_count
                for word in entry['description'].split():
                    if word not in self.stopwords and word not in self.product_words and float(entry['polarity']) != 0.0:
                        word = sm.map_synonyms(word)
                        current_descriptions_distrib += (word + ' ')
            try:
                mined_header = literal_eval(row['minedHeader'])
                for entry in mined_header:
                    for word in entry['aspect'].split():
                        if word not in self.stopwords and word not in self.product_words and float(entry['polarity']) != 0.0:
                            word = sm.map_synonyms(word)
                            attribute_set.add(word)
                            current_attribs_distrib += (word + ' ')
                            if word in attribute_helpful_counts:
                                attribute_helpful_counts[word] += helpful_count
                            else:
                                attribute_helpful_counts[word] = helpful_count
                    for word in entry['description'].split():
                        if word not in self.stopwords and word not in self.product_words and float(entry['polarity']) != 0.0:
                            word = sm.map_synonyms(word)
                            current_descriptions_distrib += (word + ' ')
            except:
                pass
            attribute_text_distribution.append(current_attribs_distrib)
            description_text_distribution.append(current_descriptions_distrib)

        if full_corpus:
            ancm.get_attribute_scores() # Exporting csv with score of neg counts by attribute
        return attribute_set, attribute_text_distribution, attribute_helpful_counts, description_text_distribution
    

    def count_vectorization(self, attribute_text_distribution: list) -> tuple:
        """
        Takes a list (attribute_text_distribution) which maintains the distribution
        of attributes mentioned across customer reviews.
        Performs the sklearn CountVectorizer().fit_transform operation on this list, and stores
        the result in word_count. word_count is now a list of vectors representing which
        attributes are present in which customer reviews).

        Returns a tuple containing word_count, and the CountVectorizer object that was created.
        """
        cv = CountVectorizer()
        word_count = cv.fit_transform(attribute_text_distribution)
        return word_count, cv
    
    def calculate_tf_idf(self, word_count, cv):
        """
        Takes 2 inputs:
        - word_count, a list of vectors representing which attributes are present in which
            customer reviews (word_count was formed by the CountVectorizer)
        - cv: a CountVectorizer object

        Uses the TfidfTransformer to generate the tf-idf vectors for the customer reviews.
        Then, for each attribute, it adds the tf-idf scores across all the customer reviews
        to get the total relevance score for each attribute. These are stored in the
        dictionary total_scores.

        These values are then reformatted into a different dictionary (formatted_scores)
        so they can subsequently be converted to a pandas DataFrame (df_tf_idf_scores).
        The values in this DataFrame are sorted in descending order of total relevance
        scores (TfIdfScore), and then this sorted DataFrame is returned.
        """
        total_scores = dict()
        tf_idf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
        tf_idf_transformer.fit(word_count)
        print("Transformer done running")

        tf_idf_vectors = tf_idf_transformer.transform(word_count)
        print("Vector created")
        feature_names = cv.get_feature_names()
        print("Feature names obtained")

        for index in range(tf_idf_vectors.shape[0]):
            doc_vector = tf_idf_vectors[index]
            df_doc_vector = pd.DataFrame(doc_vector.T.todense(), index=feature_names, columns=['TfIdfScore'])
            for word, row in df_doc_vector.iterrows():
                if word in total_scores:
                    total_scores[word] += row['TfIdfScore']
                else:
                    total_scores[word] = row['TfIdfScore']
            print("Completed outer iteration")
        print("Loop completed")
        
        formatted_scores = {
            'Word': [],
            'TfIdfScore': []
        }
        for word in total_scores:
            formatted_scores['Word'].append(word)
            formatted_scores['TfIdfScore'].append(total_scores[word])
        
        df_tf_idf_scores = pd.DataFrame.from_dict(formatted_scores)
        df_tf_idf_scores = df_tf_idf_scores.sort_values(by=['TfIdfScore'], ascending=False)
        return df_tf_idf_scores

    def add_helpful_count(self, df_tf_idf_scores, attribute_helpful_counts: dict):
        """
        Takes 2 inputs:
        - df_tf_idf_scores: a pandas DataFrame with the total relevance score
            (calculated using the tf-idf algorithm) for each attribute (product feature)
        - attribute_helpful_counts: a dictionary containing the total helpful count of
            all the customer reviews in which a specific attribute was mentioned.
        
        For each row in df_tf_idf_scores (corresponding to a particular attribute), the score
        is modified by taking the original score, and adding to it the product of the original
        score and the helpful counts value for the attribute in question divided by 1000).

        The DataFrame df_tf_idf_scores now contains these modified values, and is returned.
        """
        for index, row in df_tf_idf_scores.iterrows():
            try:
                attribute = row['Word']
                df_tf_idf_scores['TfIdfScore'][index] += (df_tf_idf_scores['TfIdfScore'][index] * attribute_helpful_counts[attribute] / 1000)
            except:
                pass
        return df_tf_idf_scores
    

    def get_top_n_attributes(self, n=20):
        """
        Takes an input n, which is an integer representing how many of the top (most relevant)
        product features (attributes) we want to return.
        Gets the top n rows of the self.df_tf_idf_scores DataFrame (these are the n most
        relevant attributes since the values in the DataFrame are sorted in descending
        order of relevance).
        Exports these top n rows as a CSV file: top_twenty_attributes.csv
        """
        top_n_df = self.df_tf_idf_scores.head(n)
        top_n_df.to_csv('templates/static/data-files/top_twenty_attributes.csv')
    
    def get_complete_attribute_ranklist(self):
        """
        Exports the data in the DataFrame self.df_tf_idf_scores as a CSV file:
        attribute_ranklist_complete.csv
        """
        self.df_tf_idf_scores.to_csv('templates/static/data-files/attribute_ranklist_complete.csv')
    

    def get_rating_by_feature_attributes_result(self):
        """
        Gets the set of unique attributes mentioned by Amazon in the ratingByFeatureAttribute
        column of mined_data.csv
        Creates a pandas DataFrame with these attributes and exports it as a CSV file:
        amazon_suggested_attributes.csv
        """
        feature_attributes_col = self.data.dropna()['ratingByFeatureAttributes'].unique()
        feature_attributes_col = list(map(lambda x: literal_eval(x), feature_attributes_col))
        print(feature_attributes_col)
        feature_attributes = set()
        for row in feature_attributes_col:
            if row == dict():
                continue
            for entry in row:
                feature_attributes.add(entry)
        print(feature_attributes)
        rating_by_feature_attributes_df = pd.DataFrame.from_dict({'attribute': list(feature_attributes)})
        rating_by_feature_attributes_df.to_csv('templates/static/data-files/amazon_suggested_attributes.csv')
    

    def get_descriptions_report(self, id, product_name: str, descriptions):
        """
        Takes 3 inputs: the product id, the product name, and the descriptions used
        to describe that product with respect to different attributes (product
        features). descriptions contains the attributes and corresponding descriptive
        words used for every attribute with respect to a single product.

        Generates a dictionary (descriptions_dict) which contains keys mapping to
        lists of product ids, corresponding product names, the top relevant
        attributes for the corresponding product, the top descriptive words used
        for those attributes for the corresponding product, and the sentiment
        (polarity) scores associated with each of those attributes for the
        corresponding product.

        After populating the values from descriptions into this dictionary,
        the dictionary is converted into a pandas DataFrame, which is
        then returned.
        """
        descriptions_dict = {
            'productID': [],
            'productName': [],
            'topRelevantAttributes': [],
            'topAttributeDescriptions': [],
            'attributeScores': []
        }
        descriptions_dict['productID'].append(int(id))
        descriptions_dict['productName'].append(product_name)
        descriptions_dict['topRelevantAttributes'].append(list(descriptions.keys()))
        descriptions_dict['topAttributeDescriptions'].append(descriptions)

        attribute_scores = dict()
        for attribute in descriptions:
            attribute_score = 0
            for description in descriptions[attribute]:
                sentiment = TextBlob(description).sentiment
                attribute_score += (sentiment.polarity * (1 - sentiment.subjectivity))
            attribute_scores[attribute] = attribute_score
        
        descriptions_dict['attributeScores'].append(attribute_scores)
        return pd.DataFrame.from_dict(descriptions_dict)


    def get_product_description_set(self, id, num_attributes=10):
        """
        Takes a product id, and the number of top features to be found
        (the default value for this is 10).

        Gets all the customer reviews that are for the product
        with the given product id. Gets the distribution of attributes
        and descriptions in the customer reviews for this product.
        Calculates the relevance of the attributes (using tf-idf scores).
        The relevance of the descriptions are also calculated using
        tf-idf scores, and the top descriptions for each attribute for the
        product in question are ascetained.

        Returns a tuple, where the first value is a dictionary mapping
        attribtues against the top descriptive words used for that
        attribute for the product in question, and where the second
        value is the name of the product.
        """
        product_df = self.data.loc[self.data['productID'] == id]
        _, attribute_text_distribution, _, description_text_distribution = self.get_attribute_set(product_df)
        word_count, cv = self.count_vectorization(attribute_text_distribution)
        df_tf_idf_scores = self.calculate_tf_idf(word_count, cv)
        top_attributes = df_tf_idf_scores.head(num_attributes)
        top_attributes_set = set(top_attributes['Word'].unique())

        descriptions = dict()
        top_descriptions = dict()
        for attribute in top_attributes_set:
            descriptions[attribute] = set()
            top_descriptions[attribute] = []

        for _, row in product_df.iterrows():
            mined_text = literal_eval(row['minedText'])
            try:
                mined_text += literal_eval(row['minedHeader'])
            except:
                pass
            for entry in mined_text:
                for attribute in top_attributes_set:
                    if attribute == entry['aspect']:
                        descriptions[attribute].add(entry['description'])
        
        descrip_word_count, descrip_cv = self.count_vectorization(description_text_distribution)
        descrip_df_tf_idf_scores = self.calculate_tf_idf(descrip_word_count, descrip_cv)
        descrip_df_tf_idf_scores = descrip_df_tf_idf_scores.sort_values(by=['TfIdfScore'], ascending=False)
        
        for _, row in descrip_df_tf_idf_scores.iterrows():
            for attribute in top_attributes_set:
                if len(top_descriptions[attribute]) < 10 and row['Word'] in descriptions[attribute]:
                    top_descriptions[attribute].append(row['Word'])
        return top_descriptions, product_df.iloc[0]['productName']


    def get_products_descriptions(self) -> None:
        """
        Generates a pandas DataFrame (complete_descriptions_df), which contains
        information, for each product, on the product id, the product name,
        the top relevant product features for that product, the top
        descriptions used for each of those attributes for the product in question,
        and the sentiment (polarity) score for each of the attributes with respect
        to the given product.

        After calculating the required values and populating this DataFrame, it is
        exported as a CSV file: product_attribute_descriptions_report.csv
        """
        product_ids = set(self.data['productID'].unique())
        complete_descriptions_df = pd.DataFrame.from_dict({
            'productID': [],
            'productName': [],
            'topRelevantAttributes': [],
            'topAttributeDescriptions': [],
            'attributeScores': []
        })
        for id in product_ids:
            descriptions, product_name = self.get_product_description_set(id)
            descriptions_df = self.get_descriptions_report(id, product_name, descriptions)
            complete_descriptions_df = pd.concat([complete_descriptions_df, descriptions_df])
        complete_descriptions_df.to_csv('templates/static/data-files/product_attribute_descriptions_report.csv')
    

    def generate_report_data(self) -> None:
        """
        Calls various functions to generate all the results (CSV files) that are required to be generated.
        """
        self.attribute_set, self.attribute_text_distribution, attribute_helpful_counts, _ = self.get_attribute_set(data=self.data, full_corpus=True)
        self.word_count, cv = self.count_vectorization(self.attribute_text_distribution)
        print("Count Vectorization complete")
        self.df_tf_idf_scores = self.calculate_tf_idf(self.word_count, cv)
        print("tf-idf calculated")
        self.df_tf_idf_scores = self.add_helpful_count(self.df_tf_idf_scores, attribute_helpful_counts)
        print("Helpful counts added")
        self.df_tf_idf_scores = self.df_tf_idf_scores.sort_values(by=['TfIdfScore'], ascending=False)
        print("values sorted")

        self.get_top_n_attributes()
        self.get_complete_attribute_ranklist()
        self.get_products_descriptions()
