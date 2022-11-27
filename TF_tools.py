import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from nltk.corpus import stopwords


def get_word_set(corpus, return_value='list'):
    words_set = set()
    for doc in corpus:
        if isinstance(doc, str):
            words = doc.split(' ')
            words_set = words_set.union(set(words))
    words_set.remove('')
    res = []
    for w in words_set:
        res.append(w)
    res = sorted(res)
    res = np.array(res)
    if return_value == 'list':
        return res
    elif return_value == 'set':
        return words_set
    else:
        print("look at the documentation for correct usage of the return_value parameter")
        return


def create_lookup(corpus):
    hashtable = {}
    list_ = get_word_set(corpus)
    set_ = get_word_set(corpus, 'set')
    for i in range(len(list_)):
        hashtable[list_[i]] = i
    return hashtable


def TF_matrix(corpus, words_set, return_type='list'):

    x = corpus
    n_docs = len(x)  # ·Number of documents in the corpus
    n_words_set = len(words_set)

    if return_type == 'list':
        df_tf = np.zeros((n_docs, n_words_set))
        table = create_lookup(x)
        for i in range(n_docs):
            words = x[i].split(' ')  # Words in the document
            for w in words:
                if w == '':
                    continue
                else:
                    w_index = table[w]
                    df_tf[i][w_index] = df_tf[i][w_index] + (1 / len(words))

        return df_tf

    elif return_type == 'df':
        df_tf = pd.DataFrame(np.zeros((n_docs, n_words_set)),
                             columns=sorted(words_set))
        for i in range(n_docs):
            words = x[i].split(' ')  # Words in the document
            for w in words:
                if w == '':
                    continue
                else:
                    df_tf[w][i] = df_tf[w][i] + (1 / len(words))
        return df_tf
    else:
        print("look at the documentation for correct usage of the return_type parameter")
        return


def get_IDF(corpus, words_set):
    x = corpus
    n_docs = len(x)
    n_words_set = len(words_set)
    idf = {}
    for w in words_set:
        k = 1    # number of documents in the corpus that contain this word
        for i in range(n_docs):
            if w in x[i].split():
                k += 1

        idf[w] = np.log10((n_docs / k-1))
    return idf


def TF_IDF(corpus, words_set, df_tf, df_type='list'):
    x = corpus
    n_docs = len(x)
    n_words_set = len(words_set)
    idf = get_IDF(x, words_set)
    df_tf_idf_ours = df_tf.copy()

    if df_type == 'list':
        table = create_lookup(x)
        for w in words_set:
            for i in range(n_docs):
                w_index = table[w]
                df_tf_idf_ours[i][w_index] = df_tf[i][w_index] * idf[w]
        return df_tf_idf_ours

    elif df_type == 'df':
        table = create_lookup(x)
        for w in words_set:
            for i in range(n_docs):
                df_tf_idf_ours[w][i] = df_tf[w][i] * idf[w]
        return df_tf_idf_ours
    else:
        print("look at the documentation for correct usage of the df_type parameter")
        return


def cos_sim(a, b):

    dot_product = dot(a, b)
    norm_a = fnorm(a)
    norm_b = fnorm(b)
    return dot_product / (norm_a * norm_b)


def jac_sim(a, b):
    intersection = len(list(set(a).intersection(b)))
    # print(intersection)
    union = (len(a) + len(b)) - intersection
    return float(intersection) / union


def dot(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i]*b[i]
    return res

# Function to return the Frobenius
# Norm of the given matrix


def fnorm(mat):

    # To store the sum of squares of the
    # elements of the given matrix
    sumSq = 0
    row = mat.shape[0]
    # col=mat.shape[1]
    for i in range(row):
        # for j in range(col):
        sumSq += pow(mat[i], 2)

    # Return the square root of
    # the sum of squares
    res = (sumSq) ** .5
    return round(res, 5)


def lowercase(corpus):
    return np.char.lower(corpus)


def remove_punctuation(corpus):
    symbols = ",!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    data = corpus.copy()
    for i in symbols:
        data = np.char.replace(data, i, ' ')
    np.char.replace(data, "'", "")
    return np.array(data)


def remove_stopwords(corpus):
    stop_words = set(stopwords.words('english'))
    res = []
    for x in corpus:
        words = x.split()
        new_text = ""
        for word in words:
            if word not in stop_words:
                new_text = new_text + " " + word
        res.append(new_text)
    return np.array(res)
