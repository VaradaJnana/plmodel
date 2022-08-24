import stanza
stanza.download('en')
nlp = stanza.Pipeline('en')

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from textblob import TextBlob


class ABSAModel2:

    def __init__(self, review: str):
        review = self.fix_review_format(review)
        self.review = review
        self.sentiment_classifier = SentimentIntensityAnalyzer()
        self.word_maps = dict()
        sentence_list = self.tokenize_sentences(review)
        pos_tagged_lists = self.tokenizing_and_pos_tagging(sentence_list)
        sentence_word_list = self.compounds_and_negatives(pos_tagged_lists)
        final_sentence_list = self.get_sentence_list_from_sentence_word_list(sentence_word_list)
        tagged_sentences = self.stopword_removal_and_pos_tagging(self.tokenize_sentence_list(final_sentence_list))
        sentence_dependencies = self.get_sentence_dependencies(final_sentence_list, sentence_word_list)
        feature_list = self.select_attribute_sublists(tagged_sentences)
        feature_clusters = self.identify_descriptive_words(feature_list, sentence_dependencies)
        final_features = self.get_final_features(feature_list, feature_clusters)
        self.result = self.format_features(final_features) # This is the field that would be picked up by the Pipeline

    

    def fix_review_format(self, review: str) -> str:
        """
        Takes a customere review (review).
        Fixes the formated of the customer review: ensures that there are spaces after
        punctuations, and adds puntuation at the end of reviews where there isn't any.
        Returns the customer review with this fixed formatting.
        """
        fixed = ""
        for char in review:
            if char == '.':
                fixed += '. '
            else:
                fixed += char
        fixed = fixed.strip()
        if fixed[-1] not in {'.', '!', '?'}:
            return fixed + '.'
        return fixed

    def tokenize_sentences(self, review: str):
        """
        Takes a customer review (review).
        Tokenizes the separate sentences in the review.
        Returns these tokenized sentences.
        """
        return sent_tokenize(review)
    
    def tokenizing_and_pos_tagging(self, sentence_list) -> list:
        """
        Takes a list of the sentences in a customer review (sentence_list).
        Tokenizes the words within each sentence of the customer review, and
        returns a list containing the Part of Speech tags of each of these
        words.
        """
        return [nltk.pos_tag(word_tokenize(sentence)) for sentence in sentence_list]
    
    def compounds_and_negatives(self, pos_tagged_lists) -> list:
        """
        Takes a list of Part of Speech tagged sentences from a customer
        review (pos_tagged_lists).
        Returns a list of new sentences, in which compound phrases or negation phrases
        are combined into one token.
        """
        new_sentence_list = []
        flag = False
        for sentence in pos_tagged_lists:
            sentence_length = len(sentence)
            new_sentence = []
            for i in range(sentence_length - 1):
                if sentence[i][0] in {'not', 'non'} or (sentence[i][1] == 'NN' and sentence[i+1][1] == 'NN'):
                    new_sentence.append(sentence[i][0] + sentence[i+1][0])
                    self.word_maps[sentence[i][0] + sentence[i+1][0]] = sentence[i][0] + '_' + sentence[i+1][0]
                    flag = True
                elif flag:
                    flag = False
                else:
                    new_sentence.append(sentence[i][0])
                    if i == sentence_length - 2:
                        new_sentence.append(sentence[i+1][0])
            new_sentence_list.append(new_sentence)
        return new_sentence_list
    
    def get_sentence_list_from_sentence_word_list(self, sentence_word_list) -> list:
        """
        Takes a list of sentences wherein each sentence is comprised of a list
        of words (sentence_word_list).
        Returns a new list of sentences, where the words in each sentence are
        combined into a single, properly formatted, string (rather than a list
        of words).
        """
        return [" ".join(word for word in sentence) for sentence in sentence_word_list]
    
    def tokenize_sentence_list(self, sentence_list):
        """
        Takes a list of sentences (sentence_list).
        Performs word tokenization on each of the words within that sentence.
        """
        return [word_tokenize(sentence) for sentence in sentence_list]
    
    def stopword_removal_and_pos_tagging(self, tokenized_sentences) -> list:
        """
        Takes a list of word-tokenized sentences (tokenized_sentences).
        Removes standard nltk stopwords (excluding the word 'not') from the
        sentences, and then performs Part of Speech tagging on each token
        in each sentence.
        Returns a list of sentences, where each sentence is a list of tuples,
        where each tuple contains the word and the corresponding Part of
        Speech tag
        """
        stop_words = set(stopwords.words('english'))
        stop_words.remove('not')
        sentence_list = [[word for word in sentence if word not in stop_words] for sentence in tokenized_sentences]
        return [nltk.pos_tag(sentence) for sentence in sentence_list]
    
    
    def get_sentence_dependencies(self, sentence_list, sentence_word_list) -> list:
        """
        Inputs:
        - sentence_list: a list of the sentences of the customer review, with
            compounds and negations joined as individual tokens
        - sentence_word_list: the same as sentence_list, except that instead of
            a string, each sentence is a list of words
        
        Uses the standard technique for extracting the dependency relationships in the
        text of the customer reviews using the Stanford NLP parser.
        Returns a list containing these dependency relationships
        """
        sentence_dependencies = []
        for sentence in sentence_list:
            doc = nlp(sentence)
            dependency_node = []
            try:
                for dependency_edge in doc.sentences[0].dependencies:
                    dependency_node.append([dependency_edge[2].text, dependency_edge[0].id, dependency_edge[1]])
                sentence_dependencies.append(dependency_node)
            except:
                pass
        
        for index, dependency_node in enumerate(sentence_dependencies):
            for i in range(len(dependency_node)):
                if int(dependency_node[i][1]) != 0:
                    try:
                        dependency_node[i][1] = sentence_word_list[index][int(dependency_node[i][1]) - 1]
                    except IndexError as e:
                        print("An error came up with this review:", self.review)
                        pass
            sentence_dependencies[index] = dependency_node
        return sentence_dependencies

    def select_attribute_sublists(self, tagged_sentences) -> list:
        """
        Takes a list of Part of Speech (POS) tagged sentences.
        Defines a set of possible POS values for tokens that could represent
        attributes (product features).
        For each entry in each sentence of the POS tagged sentences, the POS
        of the entry is contained withint the defined set of possible feature
        POS values, then the word in this entry is a potential feature.
        This word is added to the list feature_list.
        The list feature_list is returned after all the word tokens in all
        the sentences have been considered.
        """
        possible_feature_POS = {'NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        feature_list = []
        # categories = []
        for _, sentence in enumerate(tagged_sentences):
            for _, entry in enumerate(sentence):
                if entry[1] in possible_feature_POS:
                    feature_list.append(list(entry))
                    # categories.append(entry[0])
        return feature_list
    
    def identify_descriptive_words(self, feature_list, sentence_dependencies) -> list:
        """
        Inputs:
        - feature_list: a list of extracted product features (attributes)
        - sentence_dependencies: a list of the syntactic dependencies in the 
            customer review
        
        Defines a set of possible dependency edge values that could map a descriptive
        relationship.
        For each feature in feature_list, checks if there is a dependency edge which
        lies in this type, for which one of the words in conencted by the dependency
        edge is the feature. If these conditions are satisfied, then the word connected
        to the feature by the edge is the description.
        A list containing the feature word and the descriptive word is added to the
        list feature_clusters.
        The list feature_clusters is returned. It essentially contains pairs
        of attributes (product features) and the corresponding descriptions.
        """
        possible_edges = {'nsubj', 'acl:relcl', 'obj', 'dobj', 'agent', 'advmod', 'amod', 'neg', 'prep_of', 'acomp', 'xcomp', 'compound', 'csubj'}
        feature_clusters = []
        for feature in feature_list:
            related_words = []
            for dependency_node in sentence_dependencies:
                for triplet in dependency_node:
                    if (triplet[0] == feature[0] or triplet[1] == feature[0]) and triplet[2] in possible_edges:
                        if triplet[0] == feature[0]:
                            related_words.append(triplet[1])
                        else:
                            related_words.append(triplet[0])
            feature_clusters.append([feature[0], related_words])
        return feature_clusters
    

    def get_final_features(self, feature_list, feature_clusters) -> list:
        """
        Inputs:
        - feature_list: a list of the product features that have been
            extracted from the customer review
        - feature_clusters: a list of lists, where each inner list is a
            mapping between a product feature and the description for 
            that product feature.
        
        Filters out any mappings in feature_clusters for which the POS
        tag of the feature in feature_list does not contain the substring
        'NN' or 'VB'.

        Returns the filtered list (final_features), which is a list of lists,
        wherein each inner list is a mapping of a product feature and the
        description used for the product with respect to that feature.
        """
        feature_POS = dict()
        for feature in feature_list:
            feature_POS[feature[0]] = feature[1]
        
        final_features = []
        for entry in feature_clusters:
            if 'NN' in feature_POS[entry[0]] or 'VB' in feature_POS[entry[0]]:
                final_features.append(entry)
        return final_features
    

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
    
    def format_features(self, final_features) -> list:
        """
        Takes final_features, a list of lists, where each inner list is a
        mapping of a product feature, and the description used for the
        product with respect to this specific feature.

        Calculates the sentiment polarity and subjectivity for each 
        feature-description pair in final_features. Creates a list of
        dictionaries (features_list), where each dictionary contains
        information on a single feature ('aspect'), its description,
        its sentiment polarity, and its sentiment subjectivity.

        This list of dictionaries (freatures_list) is then returned.
        """
        features_list = []
        for feature in final_features:
            for description in feature[1]:
                description = str(description)
                if description in self.word_maps:
                    description = self.word_maps[description]
                sentiment = TextBlob(description.replace("_", " ") + ' ' + feature[0]).sentiment
                if feature[0] in self.word_maps:
                    aspect = self.word_maps[feature[0]]
                else:
                    aspect = feature[0]
                features_list.append({
                    'aspect': aspect,
                    'description': description,
                    'polarity': (sentiment.polarity + self.classify_sentiment(description.replace("_", " ") + ' ' + str(feature[0]))) / 2,
                    'subjectivity': sentiment.subjectivity,
                })
        return features_list
