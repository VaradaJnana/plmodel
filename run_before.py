from webscraper.webscraper import WebScraper

from absa_ensemble.pipeline import Pipeline
from report_results.report_results import ReportResults
from product_ranking.product_attribute_ranking import ProductAttributeRankingCSVGenerator

from topic_modelling.topic_modelling_searching_bertopic import TopicModellingSearchingBERTopic
from topic_modelling.topic_modelling_searching_top2vec import TopicModellingSearchingTop2Vec
from topic_modelling.improvement_extractor import ImprovementExtractor

from mindmap_generator.mindmap_generator import MindmapGenerator


class RunBefore:
    def __init__(self, product_name: str):
        """
        Takes the name of the type of products for which the dashboard
        is going to be generated (ex: exercise bike).
        Runs code from a plethora of other files to generate intermediate
        results that are required to render the dashboard, and serve
        results to the queries users might make through the dashboard
        """
        WebScraper() # scraping reviews from product links given by the user

        Pipeline() # running ABSA models on reviews to extract attributes and descriptions
        ReportResults(product_name=product_name) # generating report results (intermediate
        # results for the dashboard to be generated)
        ProductAttributeRankingCSVGenerator() # Ranking products based on attributes

        filepath="templates/static/data-files/review_data.csv"
        # Creating topic modelling models
        TopicModellingSearchingBERTopic(filepath=filepath, run_repl=False, model_already_trained=False)
        TopicModellingSearchingTop2Vec(filepath=filepath, models_already_trained=False, run_repl=False)

        ImprovementExtractor() # Extracting market improvement areas
        MindmapGenerator(product_name) # Generating mindmap image