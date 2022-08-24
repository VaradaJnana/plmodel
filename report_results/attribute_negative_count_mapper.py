import pandas as pd

class AttributeNegativeCountMapper:

    def __init__(self):
        self.attribute_score_by_product = dict()
        self.attribute_scores = {'attribute': [], 'score': []}
    
    def map_entry(self, word: str, entry: dict, row) -> None:
        """
        Takes 3 inputs:
        - word: a string, the current attribute being considered
        - entry: a dict, a specific entry of mined text, containing an attribute,
            a description, and a polarity (sentiment) score.
        - row: an entire row of data from the 'mined_data.csv' file

        Updates the count of how many times the attribute held in 'word' has been
        mentioned, and also the net polarity associated with that attribute.
        These scores are updated with respect to the specific product for which
        this row of data is based.
        These scores are updated in the dictionary self.attribute_score_by_product.
        """
        product_name = row['productName']
        polarity = float(entry['polarity'])
        if word in self.attribute_score_by_product:
            if product_name in self.attribute_score_by_product[word]:
                curr_polarity = self.attribute_score_by_product[word][product_name][0]
                curr_count = self.attribute_score_by_product[word][product_name][1]
                self.attribute_score_by_product[word][product_name] = (curr_polarity + polarity, curr_count + 1)
            else:
                self.attribute_score_by_product[word][product_name] = (polarity, 1)
        else:
            self.attribute_score_by_product[word] = dict()
            self.attribute_score_by_product[word][product_name] = (polarity, 1)
    

    def export_data(self) -> None:
        """
        Converts the data stored in the dictionary self.attribute_scores into a
        pandas DataFrame. Then exports this DataFrame as a CSV file, titled
        attribute_negative_counts.csv
        """
        pd.DataFrame.from_dict(self.attribute_scores).to_csv("templates/static/data-files/attribute_negative_counts.csv")
    
    def get_score(self, product_polarities: dict) -> int:
        """
        Takes a dictionary (product_polarities) of polarity scores (sentiment scores)
        for each product with respect to a certain attribute.
        Returns the number of products for which the attribute is mentioned at least 5
        times and where the net sentiment score (polarity score) is negative.
        """
        neg_count = 0
        for product_name in product_polarities:
            polarity, count = product_polarities[product_name]
            if polarity < 0.0 and count >= 5:
                neg_count += 1
        return neg_count
    
    def get_attribute_scores(self) -> None:
        """
        Gets the negative count scores for each of the attributes that are present as keys
        in the dictionary self.attribute_score_by_product. These scores are sorted in
        decreasing order and stored in the list attribute_scores.
        The set of attributes for which the score is greater than 0 is added to the 
        dictionary self.attribute_scores (in a way that preserves the decreasing order
        with respect to scores).

        The data in self.attribute_scores is exported as a CSV file: attribute_negative_counts.csv
        """
        attribute_scores = []
        for attribute in self.attribute_score_by_product:
            attribute_scores.append((attribute, self.get_score(self.attribute_score_by_product[attribute])))
        attribute_scores.sort(key=lambda x: x[1], reverse=True)

        for attribute, score in attribute_scores:
            if score > 0:
                self.attribute_scores['attribute'].append(attribute)
                self.attribute_scores['score'].append(score)
        self.export_data()