# unikob-comment-classifier
A classifier that classifies comments and non-comments.


## Requirements
Python 2 is used but of course switching to Python 3 is possible with minor changes.

Libraries in use:
``nltk``, ``numpy``, ``pandas``, ``sklearn``.

## Usage
1. Put data files ``native_comments.csv`` and ``area_without_comments.csv`` into ``data/`` folder.
Each data file should have csv format with each line containing an index number and a comment or non-comment.

2. Run ``main.py`` to get the classification model and see the performance.

3. Save the model as pkl file for later use (optional).

## Technical details
For any given article, the pieces of text in the comment section are classified by a binary classifier to distinguish between comment and non-comment data and to filter out the comments. Before the classification is done, the text chunks are pre-processed. Firstly, the text chunks are tokenized, and the punctuation is removed. Secondly, domain-specific stop words are removed. These ones are words, which appear often in comment sections. After the preprocessing, a trained random forest classifier is used to retrieve the comment text with associated meta data.

## Main contributors
[Jun Sun](https://github.com/yfiua), [Nico Daheim](https://github.com/ndaheim).
