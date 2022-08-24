import pandas as pd

import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')


class FeatureClassifier:

    def __init__(self):
        base_words = {
            "Style": {"style", "form", "shape", "line", "color", "colour", "tone", "space", "texture", "design", "taste", "smell", "look"},
            "Experience": {"experience", "install", "assemble", "service", "support", "help", "recommend", "problem"},
            "Quality": {"quality", "build"},
            "Price": {"price", "cost", "value", "buy", "money", "purchase", "fee", "payment", "rate"}
        }
        top_twenty_features_filepath = "templates/static/data-files/top_twenty_attributes.csv"

        feature_class_synonyms = self.get_feature_class_synonyms(base_words)
        top_features = self.get_top_features(top_twenty_features_filepath)
        classified_features = self.classify_features(top_features, base_words, feature_class_synonyms)
        self.result = (classified_features, top_features)


    def get_synonyms(self, word: str) -> set:
        """
        Takes a word (str).
        Returns a set of the synonyms of that word, derived from nltk's wordnet.
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for name in syn.lemma_names():
                synonyms.add(name.replace("_", " ").lower())
        return synonyms
    
    def get_feature_class_synonyms(self, base_words: dict) -> dict:
        """
        Takes a dictionary (base_words) of the base words contained within each of the 4 archetypal
        product feature types.
        Generates a dictionary (feature_class_synonyms), where each key is one of the archetypal
        product feature types (the same as the keys in base_words). Each of these keys map to a set
        which contains the synonyms of each of the words in the set that the key maps to in base_words.
        This dictonary (feature_class_synonyms) is returned.
        """
        feature_class_synonyms = {"Style": set(), "Experience": set(), "Quality": set(), "Price": set()}
        for feature_class in base_words:
            classname_synonyms = self.get_synonyms(feature_class.lower())
            for syn in classname_synonyms:
                feature_class_synonyms[feature_class].add(syn)
            for base_word in base_words[feature_class]:
                syns = self.get_synonyms(base_word.lower())
                for syn in syns:
                    feature_class_synonyms[feature_class].add(syn)
        return feature_class_synonyms
    

    def load_data(self, filepath: str):
        """
        Takes a filepath.
        Returns a pandas DataFrame which contains the content read in from the CSV file
        at the given filepath
        """
        return pd.read_csv(filepath)
    
    def get_top_features(self, top_twenty_features_filepath: str) -> set:
        """
        Takes the filepath to the top_twenty_features.csv file.
        Extracts all of the features from that file, and filters out the ones that are
        contained in a set of stopwords (the stopwords are defined within this function).

        Each of the remaining features are reformatted by being converted to lowercase, and
        having any underscores replaced with whitespaces.
        These features (in the format of strings) are then stored in a set (top_features),
        which is then returned.
        """
        top_twenty_features = self.load_data(top_twenty_features_filepath)
        stopwords = {"product", "good", "bad", "worse", "worst", "need", "want", "provide"}

        top_features = set()
        for entry in set(top_twenty_features['Word'].unique()):
            if entry not in stopwords:
                top_features.add(entry.lower().replace("_", " "))
        return top_features
    

    def contained_within(self, feature: str, feature_class: str, base_words: dict, feature_class_synonyms: dict) -> bool:
        """
        Takes 4 inputs:
        - feature: a string, representing the product feature (from the top 20 features) currently
                    being analyzed
        - feature_class: a string, representing which of the archetypal product feature classes
                    is currently being considered
        - base_words: a dictionary, containing the base_words defined as being contained within each of the 4
                    archetypal product feature types
        - feature_class_synonyms: a dictionary, containing the synonyms of each of the base words defined
                    for each of the archetypal product feature types
        
        Returns True if the feature in question is contained within the synonyms of the base_words mapped to by
        feature_class, or if the feature is contained within any of the base words mapped to by feature_class,
        or if any of the base_words mapped to by feature_class is contained within the feature.
        If none of these conditions are met, then returns False.
        """
        if feature in feature_class_synonyms[feature_class]:
            return True
        for base_word in base_words[feature_class]:
            if base_word in feature or feature in base_word:
                return True
        return False
    
    
    def classify_features(self, top_features: set, base_words: dict, feature_class_synonyms: dict) -> dict:
        """
        Takes 3 inputs:
        - top_features: a set, containing the filtered and properly formatted top product features
                (these are the product features that need to be classified into one of the 4
                archetypal product feature classes)
        - base_words: a dictionary, containing the base_words defined as being contained within each of the 4
                archetypal product feature types
        - feature_class_synonyms: a dictionary, containing the synonyms of each of the base words defined
                for each of the archetypal product feature types
        
        This function classifies each of the features in top_features into one of the 4 archetypal product
        feature types. A dictionary is maintained (classified_features), which maps each of these
        archetypal product feature types against the product features that were classified as belonging
        to that archetype.
        This dictionary (classified_features) is returned.
        """
        classified_features = {"Style": [], "Experience": [], "Quality": [], "Price": []}
        for feature in top_features:
            done = False
            for feature_class in base_words:
                if feature in base_words[feature_class]:
                    classified_features[feature_class].append(feature)
                    done = True
                    break
            if not done:
                for feature_class in ["Style", "Price", "Experience"]:
                    if self.contained_within(feature, feature_class, base_words, feature_class_synonyms):
                        classified_features[feature_class].append(feature)
                        done = True
                        break
                if not done:
                    classified_features["Quality"].append(feature)
        return classified_features