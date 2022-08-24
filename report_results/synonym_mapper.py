import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import PorterStemmer

import spacy
nlp = spacy.load('en_core_web_sm')

import string

class SynonymMapper:

    def __init__(self):
        self.words_added = set()
        self.synonym_mappings = dict()
        self.stemmer = PorterStemmer()
    
    def add_synonym_mappings(self, word: str) -> None:
        """
        Takes a word. Adds a mapping from each of the synonyms of the word to the word itself
        (these mappings are added to the dictionary self.synonym_mappings)
        """
        for syn in wordnet.synsets(word):
            for name in syn.lemma_names():
                self.synonym_mappings[name.replace("_", "")] = word
    
    def map_synonyms(self, word: str) -> str:
        """
        Takes a word.
        Returns the word itself if the word has already been added before. If the word has not
        been added before, then either the stemmed form of the word or one of its synonyms are
        returned, if any of those have been added before.
        If none of these conditions are met, then the word itself is returned, and mappings
        are added from the stemmed form of the word to the word itself, and from each of the
        synonyms of the word to the word itself.
        """
        stemmed = self.stemmer.stem(word)
        if word in string.punctuation:
            return word
        if word in self.words_added:
            self.synonym_mappings[stemmed] = word
            return word
        if word in self.synonym_mappings:
            self.synonym_mappings[stemmed] = self.synonym_mappings[word] # Not 100% sure about this line
            return self.synonym_mappings[word]
        
        if stemmed in self.words_added:
            return stemmed
        if stemmed in self.synonym_mappings:
            return self.synonym_mappings[stemmed]
        self.words_added.add(word)
        self.add_synonym_mappings(word)
        self.synonym_mappings[stemmed] = word
        return word