from topic_modelling.topic_modelling_searching_bertopic import TopicModellingSearchingBERTopic
from topic_modelling.topic_modelling_searching_top2vec import TopicModellingSearchingTop2Vec


class TopicModellingSearchingEnsemble:

    def __init__(self, filepath: str="templates/static/data-files/review_data.csv", speed: str="deep-learn", run_repl: bool=True, model_already_trained: bool=True, make_general_fig: bool=True):
        # os.environ["TFHUB_CACHE_DIR"] = "/var/folders/cn/dtb98nld0j7g5gfysyrv8y3m0000gp/T/tfhub_modules/063d866c06683311b44b4992fd46003be952409c"
        self.filepath = filepath
        self.speed = speed
        print("Model already trained:", model_already_trained)
        
        if make_general_fig:
            TopicModellingSearchingBERTopic(filepath=filepath, run_repl=False, model_already_trained=model_already_trained, generate_general_fig=True)
        # Now, the general figure that gets created by the BERTopic model is ready, and the model has been trained
        # (if it was not trained already)
        print("Starting with Top2Vec")
        if not model_already_trained:
            TopicModellingSearchingTop2Vec(filepath=filepath, speed=speed, models_already_trained=False, run_repl=False)
        # Now, the Top2Vec models have been trained (if they were not trained already)

    
    def merge_doc_search_result(self, bert_result, tv_result, target: int) -> list:
        """
        Takes 3 inputs:
        - bert_result: a list of strings, the top customer search results from the
            BERTopic model
        - tv_result: a list of strings, the top customer search results from the
            Top2Vec model
        - target: an int, the total number of customer reviews to be returned

        Takes the results from bert_result. Adds results from tv_result to the 
        results from bert_result until the target number of customer reviews is
        reached.
        Returns this merged list of results.
        """
        result_set = set(bert_result)
        result = list(bert_result)
        count = 0
        for entry in tv_result:
            if count == target:
                break
            if entry[1] not in result_set:
                result.append(entry[1])
                count += 1
        return result
