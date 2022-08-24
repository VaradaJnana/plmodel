# Topic Modelling Documentation

There are 6 files in this directory that contain code, and they can be split into 2 main categories:
1. Topic Modelling Code
    - `topic_modelling_results.py`
        <br>
        _Dependencies:_
        - [`matplotlib`](https://matplotlib.org/)
        - [`wordcloud`](https://pypi.org/project/wordcloud/)
        - [`bertopic`](https://pypi.org/project/bertopic/)
    - `topic_modelling_searching_bertopic.py`
        <br>
        _Dependencies:_
        - [`os`](https://docs.python.org/3/library/os.html) (Note: `os` does not need to be installed; it comes with `python` with default)
        - [`pandas`](https://pandas.pydata.org/)
    - `topic_modelling_searching_ensemble.py`
    - `topic_modelling_searching_top2vec.py`
        <br>
        _Dependencies:_
        - [`os`](https://docs.python.org/3/library/os.html) (Note: `os` does not need to be installed; it comes with `python` with default)
        - [`pandas`](https://pandas.pydata.org/)
        - [`matplotlib`](https://matplotlib.org/)
        - [`top2vec`](https://pypi.org/project/top2vec/)
        - [`nltk`](https://www.nltk.org/)
        - [`string`](https://docs.python.org/3/library/string.html) (Note: `string` does not need to be installed; it comes with `python` by default)
    - `train_bertopic_model.py`
        <br>
        _Dependencies:_
        - [`bertopic`](https://pypi.org/project/bertopic/)
2. Improvement Extraction Code
    - `improvement_extractor.py`
        <br>
        _Dependencies:_
        - [`pandas`](https://pandas.pydata.org/)
        - [`math`](https://docs.python.org/3/library/math.html) (Note: `math` does not need to be installed; it comes with `python` by default)
        - [`spacy`](https://spacy.io/)

There is also a file `__init__.py` which contains no code; it is used to designate this directory as a package, so that the classes within the 6 aforementioned code files can be imported from other directories of this project.

<hr>
<br>

<h2>Topic Modelling Code</h2>

The code in these files is used to train the topic modelling models, and then also to fetch search results using these models.

1. `train_bertopic_model.py`: Trains and saves a topic modelling model based on the BERTopic architecture. The model is saved within this directory, and the name of the model is `bertopic_model`.
2. `topic_modelling_searching_bertopic.py`: Contains functions that enable searching for customer reviews using search queries, and also generating wordclouds of relevant words based on search queries. A lot of the code that actually performs these tasks is located within `topic_modelling_results.py`, and the functions in this file are called from within `topic_modelling_searching_bertopic.py`.
3. `topic_modelling_results.py`: Contains functions with the actual code that is used to search for customer reviews or generate wordclouds using the bertopic_model based on a search query.
4. `topic_modelling_searching_top2vec.py`: Contains the code to train the topic modelling model based on the Top2Vec architecture. The model is saved within this directory as `main_model`. The file also contains code to enable searching for customer reviews using the Top2Vec model based on a search query.
5. `topic_modelling_searching_ensemble.py`: Contains the code to merge the customer review search results returned by the BERTopic model and the Top2Vec model, so as to get the overall desired number of customer reviews.

<hr>
<br>

<h2>Improvement Extraction Code</h2>

The only file that directly has anything to do with improvement extraction is `improvement_extractor.py`. However, this file has been included within the `topic_modelling` directory since `improvement_extractor.py` makes heavy use of the topic modelling architecture and models, and the code in `improvement_extractor.py` calls upon the functions defined within the files that are dedicated to topic modelling purposes.

The code in this file identifies the product features for which at least 20% of the products have a negative net sentiment score. Out of the product features that satisfy this, upto the top 10 are taken (top 10 in terms of the largest number of products for which the net sentiment score is negative). These product features are then classified as improvement areas. Information on customer reviews for these product features is extracted using the topic modelling customer searching functions, and this data is then stored in a CSV file: `improvement_areas.csv`