import contractions
import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')


class DataLoader:

    def __init__(self, filepath: str):
        self.data = None
        self.filepath = filepath

        self.load_data()
        self.write_data()


    def get_csv_data(self) -> None:
        """
        Reads the data from the CSV file present at the filepath
        stored in self.filepath.
        Reads the data into a pandas DataFrame and stores the DataFrame
        in self.data
        """
        self.data = pd.read_csv(self.filepath)


    def get_data(self):
        """
        Checks if the file at the filepath stored in self.filepath is the
        file to a CSV file. If it is, then it reads the data from the CSV
        file present at this filepath into a pandas DataFrame and stores
        the DataFrame in self.data
        If the filepath is not to a CSV file, then it raises a TypeError.
        """
        if self.filepath[-3:].lower() == 'csv':
            self.get_csv_data()
        else:
            raise TypeError("The entered data file should either be a .csv file")
    

    def to_lower_case(self, text: str) -> str:
        """
        Takes a customer review (text). Returns the lowercase version of the string.
        """
        return text.lower()

    def expand_contractions(self, text: str) -> str:
        """
        Takes a customer review (text). Expands any contractions in this customer
        review (ex: don't -> do not).
        Returns the customer review with expanded contractions.
        """
        expanded = ""
        for word in text.split():
            expanded += (contractions.fix(word) + ' ')
        return expanded
    
    def lemmatize_text(self, text: str) -> str:
        """
        Takes a customer review (text). Lemmatizes each of the words in the
        customer reviews. Returns the resulting text.
        """
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    
    def preprocess_data(self, text: str) -> str:
        """
        Takes a customer review (text). Returns the preprocessed version of the
        text. Preprocessing steps include:
        - conversion to lowercase
        - expansion of contractions
        - lemmatization
        """
        text = self.to_lower_case(text)
        text = self.expand_contractions(text)
        text = self.lemmatize_text(text)
        text = self.to_lower_case(text)
        return text

    def load_data(self) -> None:
        """
        Loads the data in the CSV file with the scraped customer reviews into
        a pandas DataFrame.
        Applies preprocessing on all of the scraped review data.
        Stores this data in self.data
        """
        self.get_data()
        self.data['reviewText'] = self.data['reviewText'].apply(lambda x: self.preprocess_data(str(x)))
        try:
            self.data['reviewHeader'] = self.data['reviewHeader'].apply(lambda x: self.preprocess_data(str(x)))
        except Exception:
            pass
        print("Data has been loaded")
    
    def write_data(self) -> None:
        """
        Takes the data in the pandas DataFrame self.data
        Exports this data to a CSV file: preprocessed_dataset.csv
        """
        self.data.to_csv('templates/static/data-files/preprocessed_dataset.csv')
