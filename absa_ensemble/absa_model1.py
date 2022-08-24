import spacy
nlp = spacy.load("en_core_web_sm")
from textblob import TextBlob

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

class ABSAModel1:

    def __init__(self, review: str):
        aspects = self.mine_aspects(review)
        self.sentiment_classifier = SentimentIntensityAnalyzer()
        self.result = self.find_sentiment(aspects)
        print("ABSA v1 review complete")


    def get_children(self, token) -> list:
        """
        Takes a token from the spacy nlp document for a customer
        review.
        Returns a list of all the syntactic children of the token.
        """
        return [child for child in token.children]
    
    
    def is_negation(self, token) -> bool:
        """
        Takes a token from the spacy nlp document for a customer
        review. Returns True if the token is a negation (if the
        syntactic dependency of the token is 'neg').
        Returns False otherwise.
        """
        return token.dep_ == 'neg'
    
    def negation_present(self, doc) -> bool:
        """
        Takes a customer review loaded into spacy's nlp model.
        Returns True if a negation word is present somewhere in
        the customer review. Returns False otherwise.
        """
        for token in doc:
            if self.is_negation(token):
                return True
        return False
    
    def get_negations(self, doc) -> dict:
        """
        Takes a customer review loaded into spacy's nlp model (doc).
        Returns a dictionary (negations) which maps negation tokens in doc
        to the tokens with which they are associated (i.e. maps the negation token
        against its syntactic parent and syntactic sibling tokens)
        """
        if self.negation_present(doc):
            negations = dict()
            for token in doc:
                children = self.get_children(token)
                for child in children:
                    if self.is_negation(child):
                        negations[child] = {token}
                        for other_child in set(children).difference({child}):
                            negations[child].add(other_child)
            return negations
        else:
            return dict()
    
    def get_token_negation(self, token, negations):
        """
        Takes a token from the spacy nlp document for a customer review.
        Also takes a dictionary (negations) mapping negation tokens to the tokens with
        which it is related (its syntactic parent and siblings).
        Returns the negation token associated with a given token if it exists.
        Returns None if token is not longuistically related to any negations in doc
        (i.e. returns None if token has not been negated in the customer review)
        """
        for negation in negations:
            if token in negations[negation]:
                return negation
        return None
    

    def get_token_compound(self, token):
        """
        Takes a token from the spacy nlp document for a customer review.
        If the token is not a part of a linguistic compound (a phrase that
        consists of more than one word), then returns None.
        If the token is a part of a linguistic compound, then returns the
        second token which forms a part of this compound.
        """
        for child in self.get_children(token):
            if child.dep_ == 'compound':
                return child
        return None
    

    def get_text(self, token, add_front, frontmost=None) -> str:
        """
        Takes 2 (or 3) tokens:
        - token: the first token
        - add_front: the token whose text is to be added in front of token
        - frontmost: the token whose text is to be added right in front
            (if it is not None)
        
        Adds teh text from add_front (if it is not None) to the front of the
        text of token. Also adds the text of frontmost (if it is not None)
        in front of the resulting text.
        Returns the final resulting text (string)
        """
        if add_front == None:
            text = token.text
        else:
            text = add_front.text + ' ' + token.text
        if frontmost == None:
            return text
        return frontmost.text + ' ' + text

    
    def get_children_nouns_verbs(self, children: list, descriptive_token, descriptive_token_text, \
        negations, negation, verb_token=None) -> tuple:
        """
        Inputs:
        - children: a list of children tokens of the token being considered
        - descriptive_token: the descriptive token currently being considered
        - descriptive_token_text: the text of the descriptive token, after adding
            any negations associated with it to the text
        - negations: a dictionary mapping negation tokens agianst the tokens with 
            which they are related
        - negation: the negation that is associated with descriptive_token, if any
        - verb_token: the verb token that is part of the description, if any

        Returns a tuple containing two values:
        1. A list of dictionaries, where each dictinary contains a mapping of an attribute
            (aspect) and the corresponding description text. These mappings are found
            from the set of children that are provided
        2. A boolean indicating whether or not a relevant noun/verb that is being
            described by the descriptive_token was found
        """
        result = []
        flag = False
        for child in children:
            if child.pos_ == 'NOUN' or child.pos_ == 'VERB':
                comp = self.get_token_compound(child)
                child_text = self.get_text(child, comp)
                flag = True
                if negation != None:
                    result.append({'aspect': child_text, 'description': descriptive_token_text})
                else:
                    noun_neg = self.get_token_negation(child, negations)
                    if verb_token == None:
                        descriptive_token_text = self.get_text(descriptive_token, noun_neg)
                    else:
                        descriptive_token_text = self.get_text(descriptive_token, verb_token, noun_neg)
                    result.append({'aspect': child_text, 'description': descriptive_token_text})
        return result, flag
    
    def get_token_noun_verb(self, token, doc, descriptive_token, descriptive_token_text, negations, negation, verb_token=None):
        """
        Inputs:
        - token: a token from the spacy nlp model of a customer review
        - doc: the customer review loaded into spacy's nlp model
        - descriptive_token: the descriptive_token currently being considered
        - descriptive_token_text: the text of descriptive_token, combined with
            any negations that are associatwed with it
        - negations: a dictionary mapping negation tokens in doc to the tokens they are related to
        - negation: the negation token that is known to be associated with descriptive_token
            beforehand, if any
        - verb_token: the verb token that is part of the description, if any

        Returns a list (result) of dictionaries, where the dictionaries contain information
        on the attributes and corresponding descriptions for the given token.
        If the given token does not have a relevant attribute->descriptin mapping, then 
        an empty list is returned.
        """
        result = []
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            comp = self.get_token_compound(token)
            token_text = self.get_text(token, comp)
            if negation != None:
                result.append({'aspect': token_text, 'description': descriptive_token_text})
            else:
                noun_neg = self.get_token_negation(token, negations)
                if verb_token == None:
                    descriptive_token_text = self.get_text(descriptive_token, noun_neg)
                else:
                    descriptive_token_text = self.get_text(descriptive_token, verb_token, noun_neg)
                result.append({'aspect': token_text, 'description': descriptive_token_text})
        for tok in doc:
            if (tok.pos_ == 'NOUN' or tok.pos_ == 'VERB') and (token in self.get_children(tok)):
                comp = self.get_token_compound(tok)
                tok_text = self.get_text(tok, comp)
                if negation != None:
                    result.append({'aspect': tok_text, 'description': descriptive_token_text})
                else:
                    noun_neg = self.get_token_negation(tok, negations)
                    if verb_token == None:
                        descriptive_tok_text = self.get_text(descriptive_token, noun_neg)
                    else:
                        descriptive_tok_text = self.get_text(descriptive_token, verb_token, noun_neg)
                    result.append({'aspect': tok_text, 'description': descriptive_tok_text})
        return result


    def find_adjective_target(self, descriptive_token, doc, negations, negation=None):
        """
        Inputs:
        - descriptive_token: a descriptive token (i.e. a token whose syntactic dependency is 'ADJ')
        - doc: the customer review loaded into spacy's nlp model
        - negations: a dictionary mapping negation tokens in doc to the tokens they are related to
        - negation: the negation token that is known to be associated with descriptive_token
            beforehand, if any
        
        Returns a list (result) of dictionaries, where the dictionaries contain mappings
        of all attributes and the corresponding descriptions for the given
        descriptive token.
        """
        result = []
        children = self.get_children(descriptive_token)

        descriptive_token_text = self.get_text(descriptive_token, negation)

        if children != []:
            child_nouns_verbs, flag = self.get_children_nouns_verbs(children, descriptive_token, descriptive_token_text, negations, negation)
            result += child_nouns_verbs
            if not flag:
                for token in doc:
                    if descriptive_token in self.get_children(token):
                        token_noun_verb = self.get_token_noun_verb(token, doc, descriptive_token, descriptive_token_text, negations, negation)
                        if token_noun_verb != []:
                            result += token_noun_verb
                        nouns_verbs, _ = self.get_children_nouns_verbs(self.get_children(token), descriptive_token, descriptive_token_text, negations, negation)
                        result += nouns_verbs
                        return result
            return result
        else:
            for token in doc:
                if descriptive_token in self.get_children(token):
                    token_noun_verb = self.get_token_noun_verb(token, doc, descriptive_token, descriptive_token_text, negations, negation)
                    if token_noun_verb != []:
                        result += token_noun_verb
                    child_nouns_verbs, _ = self.get_children_nouns_verbs(self.get_children(token), descriptive_token, descriptive_token_text, negations, negation)
                    result += child_nouns_verbs
                    return result

    
    def get_verb_of_adverb(self, adverb_token, doc):
        """
        Inputs:
        - adverb_token: a token in doc whose syntactic dependency was 'ADV'
        - doc: the customer review loaded into spacy's nlp model

        Returns the verb token associated with the adverb_token if it exists.
        Returns None if there is no verb token linguistically related with
        the given adverb token.
        """
        for token in doc:
            if token.pos_ == 'VERB' and adverb_token in self.get_children(token):
                return token
        return None

    def find_adverb_target(self, descriptive_token, doc, negations, verb_token=None, negation=None):
        """
        Inputs:
        - descriptive_token: a descriptive token (i.e. a token whose syntactic dependency is 'ADJ')
        - doc: the customer review loaded into spacy's nlp model
        - negations: a dictionary mapping negation tokens in doc to the tokens they are related to
        - verb_token: the verb token that is part of the description, if any
        - negation: the negation token that is known to be associated with descriptive_token
            beforehand, if any
        
        Returns a list (result) of dictionaries, where the dictionaries contain mappings
        of all attributes and the corresponding descriptions for the given
        descriptive token.
        """
        result = []
        children = self.get_children(descriptive_token)

        descriptive_token_text = self.get_text(descriptive_token, verb_token, negation)

        if children != []:
            child_nouns_verbs, flag = self.get_children_nouns_verbs(children, descriptive_token, descriptive_token_text, negations, negation, verb_token)
            result += child_nouns_verbs
            if not flag:
                for token in doc:
                    if descriptive_token in self.get_children(token):
                        token_noun_verb = self.get_token_noun_verb(token, doc, descriptive_token, descriptive_token_text, negations, negation, verb_token)
                        if token_noun_verb != []:
                            result += token_noun_verb
                        nouns_verbs, _ = self.get_children_nouns_verbs(self.get_children(token), descriptive_token, descriptive_token_text, negations, negation, verb_token)
                        result += nouns_verbs
                        return result
            return result
        else:
            for token in doc:
                if descriptive_token in self.get_children(token):
                    token_noun_verb = self.get_token_noun_verb(token, doc, descriptive_token, descriptive_token_text, negations, negation, verb_token)
                    if token_noun_verb != []:
                        result += token_noun_verb
                    child_nouns_verbs, _ = self.get_children_nouns_verbs(self.get_children(token), descriptive_token, descriptive_token_text, negations, negation, verb_token)
                    result += child_nouns_verbs
                    return result
    

    def mine_aspects(self, text: str) -> list:
        """
        Takes a customer review (text).
        Returns a list of dictionaries (aspects), where each dictionary
        has a key 'aspect' which maps to an attribute (product feature),
        and a key 'description' which maps to the description used
        for that attribute.
        The list effectively contains dictionaries highlighting all
        of the attribute-description pairs in the customer review.
        """
        aspects = []
        doc = nlp(text)
        negations = self.get_negations(doc)
        for token in doc:
            if token.pos_ == 'ADJ':
                token_neg = self.get_token_negation(token, negations)
                result = self.find_adjective_target(token, doc, negations, token_neg)
                if result != None:
                    aspects += result
            elif token.pos_ == 'ADV':
                verb_token = self.get_verb_of_adverb(token, doc)
                token_neg = self.get_token_negation(token, negations)
                result = self.find_adverb_target(token, doc, negations, verb_token, token_neg)
                if result != None:
                    aspects += result
        return aspects
    

    def classify_sentiment(self, phrase: str) -> float:
        """
        Takes a phrase (string).
        Uses the VADER SentimentIntensityAnalyzer to get the sentiment
        polarity score for the phrase (classifies the sentiment of the
        phrase).
        Returns the compound polarity score of this sentiment classification
        """
        sentiment = self.sentiment_classifier.polarity_scores(phrase)
        return sentiment['compound']

    def find_sentiment(self, aspects):
        """
        Takes a list of dictionaries (aspects) where each dictionary contains
        a single mapping of an attribute and the corresponding description.

        Obtains the sentiment polarity score and the sentiment subjectivity score
        for the phrases contained within each of these attribute -> description
        mappings, and adding both of these values to the corresponding dictionary.

        Returns the aspects list after modifying each of the dictionaries within
        aspects after adding these sentiment polarity and subjectivity values
        """
        for aspect in aspects:
            sentiment = TextBlob(aspect['description'] + ' ' + aspect['aspect']).sentiment
            aspect['polarity'] = (sentiment.polarity + self.classify_sentiment(aspect['description'] + ' ' + aspect['aspect'])) / 2
            aspect['subjectivity'] = sentiment.subjectivity
        return aspects