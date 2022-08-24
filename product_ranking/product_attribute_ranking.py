import pandas as pd
from ast import literal_eval

class ProductAttributeRankingCSVGenerator:

    def __init__(self, filepath: str="templates/static/data-files/product_attribute_descriptions_report.csv"):
        data_df = self.load_data(filepath)
        self.fix_data_format(data_df)

        self.attribute_product_mapping = dict()
        self.generate_product_rankings_by_attribute(data_df)
        self.export_data()
    
    
    def load_data(self, filepath: str):
        """
        Takes a filepath
        Loads the data in the CSV file at the given filepath into a pandas DataFrame and returns it.
        """
        return pd.read_csv(filepath)
    
    def fix_data_format(self, data_df) -> None:
        """
        Takes a pandas DataFrame.
        The data in the topRelevantAttributes column is in the format of a list, but is currently stored in a
        string (this happened since the data was stored in a CSV file). Similarly, the data in the
        topAttributeDEscriptions and the attributeScores column are in the form of dictionaries, but are currently
        stored as strings.
        This function converts these strings back into the intended data type, thereby fixing the format in which 
        the data is stored in the pandas DataFrame.
        """
        data_df['topRelevantAttributes'] = data_df['topRelevantAttributes'].apply(lambda x: literal_eval(x))
        data_df['topAttributeDescriptions'] = data_df['topAttributeDescriptions'].apply(lambda x: literal_eval(x))
        data_df['attributeScores'] = data_df['attributeScores'].apply(lambda x: literal_eval(x))
    

    def get_attribute_info_for_product(self, attribute: str, df_row) -> tuple:
        """
        Takes a specific attribute (product feature).
        Also takes a row (df_row) of the pandas DataFrame which contains the data from 
        product_attribute_descriptions_report.csv
        From this row of data, extracts the product id, the product name, the relevant descriptive words (the
        words used to describe the product in this row of the DataFrame with respect to the given attribute),
        and the sentiment score for the given attribute.
        Returns a tuple containing these 4 values (in this order).
        """
        product_id = int(df_row['productID'])
        product_name = df_row['productName']
        descriptive_words = df_row['topAttributeDescriptions'][attribute]
        score = df_row['attributeScores'][attribute]
        return (product_id, product_name, descriptive_words, score)
    
    def add_attribute_product_info(self, attribute: str, attribute_info_for_product: tuple) -> None:
        """
        Takes an attribute (product feature).
        Also takes a tuple (attribute_info_for_product) which contains information on the customers'
        perceptions of a specific product with respect to the given attribute.
        If the given attribute already exists as a key in the dictionary self.attribute_product_mapping,
        then attribute_info_for_product is appended to the list mapped to by this key.
        If the given attribute does not exist as a key in the dictionary self.attribute_product_mapping,
        then this attribute is added as a key, and it maps to a list which contains attribute_product_info.
        """
        if attribute in self.attribute_product_mapping:
            self.attribute_product_mapping[attribute].append(attribute_info_for_product)
        else:
            self.attribute_product_mapping[attribute] = [attribute_info_for_product]
    
    def sort_data(self) -> None:
        """
        self.attribute_product_mapping is a dictionary which has attributes (product features)
        as keys, and lists of tuples (with information on specific products) as values.
        For each list stored as a value (in the dictionary), this function sorts that list in
        descending order based on the value stored at the third index of the tuple (i.e. in
        descending order of the sentiment scores). 
        """
        for attribute in self.attribute_product_mapping:
            self.attribute_product_mapping[attribute].sort(key=lambda x: x[3], reverse=True)
    
    def generate_product_rankings_by_attribute(self, data_df) -> None:
        """
        Takes a pandas DataFrame (data_df) which contains the information from the file 
        product_attribute_descriptions_report.csv
        For each row of data_df, this function iterates through each of the attributes stored in the 
        topRelevantAttributes column, and adds the information pertaining to that attribute from that row
        into the list mapped to by the attribute in the dictionary self.attribute_product_mapping
        Then, this function sorts each of these lists that are mapped to in decreasing order of sentiment
        scores.
        """
        for _, df_row in data_df.iterrows():
            for attribute in df_row['topRelevantAttributes']:
                self.add_attribute_product_info(attribute, self.get_attribute_info_for_product(attribute, df_row))
        self.sort_data()
    
    
    def get_final_formatted_dict(self) -> dict:
        """
        Reformats the data that is stored in the dictionary self.attribute_product_mapping.
        self.attribute_product_mapping has attributes as keys, and for each key, the corresponding value is the
        relevant product mapping.
        This function prepares a new dictionary with two keys: 'attribute' and 'productMapping'. Each of these keys
        map to a list. Each key-value pair in self.attribute_product_mapping is taken and the key is added to
        the list mapped to by 'attribute', and the value is added to the same index of the list mapped to by 
        'productMapping'.
        The resulting dictionary is designed in the correct format for being exported to a CSV file, and is returned.
        """
        data_dict = {'attribute': [], 'productMapping': []}
        for attribute in self.attribute_product_mapping:
            data_dict['attribute'].append(attribute)
            data_dict['productMapping'].append(self.attribute_product_mapping[attribute])
        return data_dict
    
    def export_data(self) -> None:
        """
        Reformats the data in the dictionary self.attribute_product_mapping into a format that is ready to 
        be written as a CSV file, and then writes a CSV file containing this data.
        This file is called attribute_product_mappings.csv
        """
        final_df = pd.DataFrame.from_dict(self.get_final_formatted_dict())
        final_df.to_csv('templates/static/data-files/attribute_product_mappings.csv')
