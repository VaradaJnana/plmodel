import pydot
from ast import literal_eval
import pandas as pd

from mindmap_generator.feature_classifier import FeatureClassifier
from mindmap_generator.table_string_creator import TableStringCreator

class MindmapGenerator():
    def __init__(self, product_name: str):
        """
        Input: product_name: a string containing the name of the general type of product
        for which the mindmap is being generated.
        For example, if the dashboard is currently being generated for a set of products
        that are all exercise bikes, then product_name should contain the string
        "exercise bike".
        """
        classified_features, top_features = FeatureClassifier().result
        attribute_product_mappings = self.load_data("templates/static/data-files/attribute_product_mappings.csv")
        feature_mapping = self.get_feature_mapping(attribute_product_mappings, top_features)
        self.create_mindmap(product_name, classified_features, feature_mapping)
    

    def fix_df_types(self, data_df):
        """
        Takes a pandas DataFrame (data_df).
        The values stored in the productMapping column of the dataFrame are in the format of lists of
        tuples, however, these have been stored as strings (since the data was stored in a CSV file).
        These values are brought back into the format of lists of tuples.
        Additionally, the strings in the attribute column are converted to lowercase, and any
        underscores are replaced with whitespaces.

        The DataFrame with these reformatted values is returned.
        """
        data_df['productMapping'] = data_df['productMapping'].apply(lambda x: literal_eval(x))
        data_df['attribute'] = data_df['attribute'].apply(lambda x: x.lower().replace("_", " "))
        return data_df

    def load_data(self, filepath: str):
        """
        Takes a filepath.
        Returns a pandas DataFrame which contains the content read in from the CSV file
        at the given filepath, after fixing the format of the data in the DataFrame.
        """
        return self.fix_df_types(pd.read_csv(filepath))
    

    def sort_entries_by_id(self, unsorted_list: list) -> list:
        """
        Takes a list of tuples, where the first value in the tuple is the product id,
        and the second value in the tuple is the product name (truncated).

        This function sorts the tuples in the list in increasing order of product ids
        (the first value in the tuple) and returns this sorted list.
        """
        return sorted(unsorted_list, key=lambda item: item[0])

    def get_good_bad_neutral_mapping(self, product_mapping: tuple) -> tuple:
        """
        Takes the product mapping (product_mapping) for a specific product feature (attribute).
        This product_mapping is in the format of a list of tuples, where each tuple contains
        information on customer perceptions on a specific product with respect to the
        product feature (attribute) in question.

        Extracts the product id, product name, and sentiment score for each of the tuples
        in the product mapping. Based on the sentiment score, adds a tuple with the product id
        and the product name to either the good, the neutral, or the bad list.

        Then, the function sorts the tuples in each of these lists into ascending order of the
        product ids. Returns a tuple containing these 3 lists.
        """
        good = []
        neutral = []
        bad = []
        for id, prod_name, _, score in product_mapping:
            if len(prod_name) > 20:
                prod_name = prod_name[:20] + "..."
            result = (id, f"{id}. {prod_name}")
            if score > 0.0:
                good.append(result)
            elif score == 0.0:
                neutral.append(result)
            else:
                bad.append(result)
        
        good = self.sort_entries_by_id(good)
        neutral = self.sort_entries_by_id(neutral)
        bad = self.sort_entries_by_id(bad)
        return good, neutral, bad
    
    def get_feature_mapping(self, attribute_product_mappings, top_features: set) -> dict:
        """
        Maps features to which products are good, bad, and neutral for that feature
        """
        feature_mapping = dict()
        for _, df_row in attribute_product_mappings.iterrows():
            if df_row['attribute'] not in top_features:
                continue
            good, neutral, bad = self.get_good_bad_neutral_mapping(df_row['productMapping'])
            feature_mapping[df_row['attribute']] = TableStringCreator(good, neutral, bad).table_string
        return feature_mapping
    

    def export_mindmap(self, graph) -> None:
        """
        Takes a graph object.
        Creates a mindmap image using the information in the graph object, and exports this image
        as the file 'mindmap.png'
        """
        graph.write_png('templates/static/images/mindmap.png')
    
    def create_mindmap(self, product_name: str, classified_features: dict, feature_mapping: dict) -> None:
        """
        Takes 3 inputs:
        - product_name: a string, containing the name of the broad type of products being generated in the
            dashboard currently (ex: "exercise bike")
        - classified_features: a dictionary classifying the top 20 features into one of 4 archetypal
            product feature types
        - feature_mapping: a dictionary mapping each of the top 20 features to a string containing a formatted
            table, which represents the products classified as being Good, Bad, and Neutral with respect to
            the product feature in question.
        
        Generates a mindmap image using the data from the inputs and exports the mindmap image as
        'mindmap.png'
        """
        graph = pydot.Dot(graph_type="digraph", rankdir="LR", overlap="prism1000")
        graph.set_node_defaults(shape="box", nojustify=True, fontname="Monospace")

        root = product_name
        # Adding first layer (broad taxonomy)
        for feature_class in classified_features:
            graph.add_edge(pydot.Edge(root, feature_class))
        
        for feature_class in classified_features:
            features = classified_features[feature_class]
            for feature in features:
                if feature not in feature_mapping:
                    continue
                graph.add_edge(pydot.Edge(feature_class, feature))
                graph.add_edge(pydot.Edge(feature, feature_mapping[feature]))
        self.export_mindmap(graph)
