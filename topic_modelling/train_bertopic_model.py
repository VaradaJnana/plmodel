from bertopic import BERTopic

class TrainBERTopicModel:

    def __init__(self, data_df, output_filepath: str):
        documents = self.generate_document_list(data_df)
        self.topic_model = None
        self.train_model(documents)
        self.save_model(str(output_filepath))


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
                documents.append(str(row['reviewHeader'] + '. ' + row['reviewText']))
            except:
                documents.append(str(row['reviewText']))
        return documents
    
    def train_model(self, documents: list) -> None:
        """
        Trains the BERTopic model on the customer reviews contained in the list
        of strings (documents)
        """
        self.topic_model = BERTopic()
        _, _ = self.topic_model.fit_transform(documents)
    
    def save_model(self, filepath: str) -> None:
        """
        Saves the BERTopic model that has been trained.
        The model is saved as "bertopic_model"
        """
        self.topic_model.save(filepath)
        print(f"Model save successful! Model saved at: {filepath}")