import pandas as pd

from absa_ensemble.data_loader import DataLoader
from absa_ensemble.absa_ensemble_model import ABSAEnsemble

class Pipeline:
    
    def __init__(self, filepath: str="templates/static/data-files/review_data.csv"):
        self.filepath = filepath
        self.preprocess_data()
        self.mine_data()
    
    def preprocess_data(self) -> None:
        """
        Runs the code from data_loader.py (i.e. runs the code to preprocess
        the customer review text)
        """
        DataLoader(self.filepath)
    
    def mine_aspects(self, df_column):
        """
        Runs the code in absa_ensemble.py on a single preprocessed
        customer review.
        Returns the mined version of this data: i.e. the results of
        running all of the internal aspect based sentiment analysis 
        models on that customer review.
        """
        return df_column.apply(lambda x: ABSAEnsemble(str(x)).result)
    
    def write_data(self, df) -> None:
        """
        Takes a pandas DataFrame (df).
        Exports the data in df to a CSV file: mined_data.csv
        """
        df.to_csv('templates/static/data-files/mined_data.csv')
    
    def mine_data(self) -> None:
        """
        Reads in the data on the preprocessed customer reviews.
        Runs the code from absa_ensemble to extract the product features
        and descriptions from the customer reviews.
        Exports the data to mined_data.csv
        """
        df = pd.read_csv('templates/static/data-files/preprocessed_dataset.csv')
        df['minedText'] = self.mine_aspects(df['reviewText'])
        try:
            df['minedHeader'] = self.mine_aspects(df['reviewHeader'])
        except KeyError:
            print("No reviewHeader column exists")
        self.write_data(df)
