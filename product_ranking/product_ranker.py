import pandas as pd
from ast import literal_eval


class ProductRanker:

    def __init__(self, attribute: str):
        """
        Takes in the attribute (product feature) for which the user wishes to rank the products.
        Computes and stores the product rankings in a pandas DataFrame in self.result.
        """
        data_df = self.load_data()
        self.result = self.get_product_rankings(attribute, data_df)

    
    def fix_data_format(self, data_df):
        """
        Takes a pandas DataFrame, which contains the data read in from the attribute_product_mappings.csv
        file.
        Currently, even though the data in the productMapping column is in the form of a list of tuples,
        the data has been stored as a string (this happened when it was stored as a CSV). The data
        in this column is reformatted back into lists of tuples (from strings).
        The resulting DataFrame is returned.
        """
        data_df['productMapping'] = data_df['productMapping'].apply(lambda x: literal_eval(x))
        return data_df

    def load_data(self):
        """
        Loads the data stored in the attribute_product_mappings.csv file into a pandas DataFrame.
        Then, it reformats some of the data from strings into the corresponding literals.
        Returns the reformatted DataFrame.
        """
        return self.fix_data_format(pd.read_csv('templates/static/data-files/attribute_product_mappings.csv'))

    
    def extract_product_mapping(self, attribute: str, data_df):
        """
        Takes the attribute (product feature) based on which the products are to be ranked.
        Also takes a pandas DataFrame (data_df), which contains the information of all the 
        attribute product mappings.
        Finds the row of the DataFrame which contains the information for the chosen
        attribute, and returns the product mapping for that row.
        If the row for this attribute does not exist, a KeyError is raised.
        """
        for _, df_row in data_df.iterrows():
            if df_row['attribute'] == attribute:
                return df_row['productMapping']
        raise KeyError("An issue occurred while fetching the entered attribute\nEntered attribute does not exist!")
    
    def generate_ranking_csv(self, product_mapping):
        """
        Takes the product mapping for the attribute based on which the products are being ranked.
        product_mapping is currently a list of tuples.
        This function prepares and returns a pandas DataFrame where each row contains the information
        from one of the tuples in product_mapping. The data in each tuple is split across four colunms, namely
        productID, productName, descriptiveWords, and score.
        """
        result = {'productID': [], 'productName': [], 'descriptiveWords': [], 'score': []}
        for product in product_mapping:
            result['productID'].append(product[0])
            result['productName'].append(product[1])
            result['descriptiveWords'].append(product[2])
            result['score'].append(product[3])
        data_df = pd.DataFrame.from_dict(result)
        return data_df

    def get_product_rankings(self, attribute: str, data_df):
        """
        Takes the attribute (product feature) based on which the products need to be ranked.
        Also takes a pandas DataFrame (data_df) which contains all of the attribute product mappings.
        Returns a pandas DataFrame which contains the product rankings based on the chosen attribute.
        """
        product_mapping = self.extract_product_mapping(attribute, data_df)
        return self.generate_ranking_csv(product_mapping)
