import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.externals import joblib


def get_features(document, word_features):
    document_words = set(document)
    features = [(word in document_words) for word in word_features]
    # number of words in document
    features.append(len(document))
    # TODO: add more features
    return features

# main
def main():
    # params
    test_size = 0.2
    n_trees = 128

    # read data
    df_c = pd.read_csv('data/native_comments.csv', header=None, encoding='utf8')
    df_nc = pd.read_csv('data/area_without_comments.csv', encoding='utf8')

    comments = df_c[1].values
    non_comments = df_nc['1'].values

    # removes punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    # tokenize
    #tokenized_c = [nltk.word_tokenize(i) for i in comments]
    #tokenized_nc = [nltk.word_tokenize(i) for i in non_comments]

    tokenized_c = [tokenizer.tokenize(i) for i in comments]
    tokenized_nc = [tokenizer.tokenize(i) for i in non_comments]

    # domain specific stop words
    stop_words = np.array(['share', 'shares', 'like', 'report', 'sign in', 'register', 'sign up', 'facebook',
                      'twitter', 'tumblr', 'reddit', 'login', 'reply', 'replies', 'flag', 'minutes ago',
                      'hours ago', 'days ago', 'months ago', 'likes', 'sort By', 'newest', 'oldest', 'follow',
                      'view all comments', 'recommendations', 'loading comments'])

    # split into training and test sets
    c_train, c_test = train_test_split(tokenized_c, test_size=test_size)
    nc_train, nc_test = train_test_split(tokenized_nc, test_size=test_size)

    freq_words = np.array(nltk.corpus.stopwords.words('english'))

    for i in range(10):
        freq_words = np.concatenate([freq_words, freq_words])

    # concatenate all words in the training set
    #all_words = np.concatenate(map(np.concatenate, [c_train, nc_train]))
    all_words = np.concatenate([np.concatenate(c_train), freq_words])

    #remove domain specific stop words
    all_words = np.setdiff1d(all_words, stop_words)
    np.set_printoptions(threshold=np.inf)
    word_freq = nltk.FreqDist(w.lower() for w in all_words)

    word_features = list(word_freq)[:2000]
    #joblib.dump(word_features, 'word_features.pkl')

    # get features
    X_train = [get_features(i, word_features) for i in c_train + nc_train]
    X_test = [get_features(i, word_features) for i in c_test + nc_test]

    # ground truth
    y_train = [1] * len(c_train) + [0] * len(nc_train)
    y_test = [1] * len(c_test) + [0] * len(nc_test)

    #classifier
    clf = RandomForestClassifier(n_estimators=n_trees)
    clf.fit(X_train, y_train)
    
    v_pred = clf.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, v_pred)
    print auc

    #joblib.dump(clf, 'comment_clf.pkl')
    #joblib.dump(word_features, 'word_features.pkl')

if __name__ == '__main__':
    # init
    main()
