# ABSA Ensemble Documentation

There are 5 files within this directory that contain code:
1. `pipeline.py`
2. `data_loader.py`
3. `absa_ensemble.py`
4. `absa_model1.py`
5. `absa_model2.py`

The files in this directory are used to preprocess the text of the customer reviews that have been webscraped, and then perform aspect based sentiment analysis on the customer reviews.

<br>

**`pipeline.py`:**

_Dependencies:_
- [`pandas`](https://pandas.pydata.org/)

This file controls the flow of all the code within this directory. It first preprocesses the data by running the code in `data_loader.py`, and then runs the aspect based sentiment analysis code (from `absa_ensemble.py`)

<br>

**`data_loader.py`:**

_Dependencies:_
- [`pandas`](https://pandas.pydata.org/)
- [`contractions`](https://pypi.org/project/contractions/)
- [`spacy`](https://spacy.io/)

The code in this file preprocesses the text of the customer reviews, and then exports the data to `preprocessed_dataset.csv`. Teh preprocessing involves conversion to lowercase, expanding contractions, and lemmatizing text.

<br>

**`absa_ensemble.py`:**

The code in this file runs the aspect based sentiment analysis code from the following two files:
1. `absa_model1.py`
    - _Dependencies:_
        - [`spacy`](https://spacy.io/)
        - [`textblob`](https://textblob.readthedocs.io/en/dev/)
        - [`nltk`](https://www.nltk.org/)
2. `absa_model2.py`
    - _Dependencies:_
        - [`stanza`](https://stanfordnlp.github.io/stanza/)
        - [`textblob`](https://textblob.readthedocs.io/en/dev/)
        - [`nltk`](https://www.nltk.org/)

It then merges the entries of the attributes and descriptions extracted by the models in these two files.