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

## Main contributors
[Jun Sun](https://github.com/yfiua), Nico Daheim(https://github.com/ndaheim).
