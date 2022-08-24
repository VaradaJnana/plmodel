import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from wordcloud import WordCloud

from bertopic import BERTopic


class TopicModellingResults:

    def __init__(self, model_filepath: str):
        self.model = BERTopic.load(str(model_filepath))


    def get_top_topic_words_and_scores(self, model, topic_num: int):
        """
        Takes a BERTopic model.
        Also takes the number (int) of a topic.
        Returns the top words from the topic with the given topic_num (and returns
        the scores of those top words as well).
        """
        return model.get_topic(topic_num)
    
    def get_top_topic_words(self, model, topic_num: int) -> list:
        """
        Takes a BERTopic model.
        Also takes the number (int) of a topic.
        Returns a list of the top words from the topic with the given topic_num
        """
        return [entry[0] for entry in self.get_top_topic_words_and_scores(model=model, topic_num=topic_num)]
    
    def generate_wordcloud_for_query(self, model, query: str, top_n: int=5) -> None:
        """
        Takes 3 inputs:
        - model: the BERTopic model
        - query: a string, the search query for which the wordcloud is to be
            generated
        - top_n: an int, the number of topics from which the words for the wordcloud
            will be sourced
        
        Generates a wordcloud with the top words from each of the most relevant words
        from the top_n topics related to the given search query
        """
        words = []
        for i, topic_num in enumerate(model.find_topics(query, top_n=top_n)[0]):
            for word, _ in model.get_topic(topic_num):
                for _ in range(top_n - i):
                    words.append(word)
        text = " ".join(word for word in words)
        wordcloud = WordCloud(collocations=False, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"General Wordcloud 1: {query}")
        plt.axis('off')
        plt.show()
        wordcloud.to_file(f"templates/static/wordclouds/general_wordcloud_1_{query.replace(' ', '_')}.png")

    
    def generate_second_wordcloud_for_query(self, model, query: str, top_n: int=5) -> None:
        """
        Takes 3 inputs:
        - model: the BERTopic model
        - query: a string, the search query for which the wordcloud is to be
            generated
        - top_n: an int, the number of customer reviews from which the words for
            the wordcloud will be sourced
        
        Generates a wordcloud with the words from the top_n customer reviews that
        are most related to the search query.
        """
        top_reviews = self.document_search_by_query(model, query, top_n)
        text = " ".join(review for review in top_reviews)
        wordcloud = WordCloud(collocations=False, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"General Wordcloud 2: {query}")
        plt.axis('off')
        plt.show()
        wordcloud.to_file(f"templates/static/wordclouds/general_wordcloud_2_{query.replace(' ', '_')}.png")
    
    
    def document_search_by_query(self, model, query: str, top_n: int=5) -> list:
        """
        Takes 3 inputs:
        - model: the BERTopic model
        - query: a string, the search query for which the wordcloud is to be
            generated
        - top_n: an int, the number of topics from which the customer reviews
            for the sexarch query will be sourced
        
        Returns a list of the customer reviews from each of the top_n topics
        that are most relevant to the search query
        """
        result = []
        for topic_num in model.find_topics(query, top_n=top_n)[0]:
            topic_reviews = model.representative_docs[topic_num]
            result += topic_reviews
        return result
    
    def generate_topic_space_overview(self, model) -> None:
        """
        Takes a BERTopic model.
        Generates an image of the top words associated with each of the topics
        created by the BERTopic model.
        """
        fig = model.visualize_barchart(top_n_topics=len(model.get_topics()))
        fig.show()
        fig.write_image("topic_word_scores_barcharts.png")