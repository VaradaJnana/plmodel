from absa_ensemble.absa_model1 import ABSAModel1
from absa_ensemble.absa_model2 import ABSAModel2

class ABSAEnsemble:

    def __init__(self, review: str):
        res1 = ABSAModel1(review).result # Running ABSAModel1 on the customer review
        res2 = ABSAModel2(review).result # Running ABSAModel2 on the customer review
        self.result = self.merge_entries(res1 + res2) # Merges the results from the two ABSA Models
    
    def merge_entries(self, results: list) -> list:
        """
        Takes a list of dictionaries (results), which contains the results of the
        mined attributes, descriptions and sentiment scores produced by the two
        ABSA models.
        It merges the mined entries from the two models, combining the net polarity
        scores in instances in which both models picked up the same attribute-description
        pair.
        This merged list of dictionaries (merged) is returned.
        """
        merged = []
        for entry in results:
            flag = True
            entry['aspect'] = entry['aspect'].replace(' ', '_')
            entry['description'] = entry['description'].replace(' ', '_')
            for i, processed in enumerate(merged):
                if processed['aspect'] == entry['aspect'] and processed['description'] == entry['description']:
                    flag = False
                    merged[i]['polarity'] = merged[i]['polarity'] + entry['polarity']
            if flag:
                merged.append(entry)
        return merged