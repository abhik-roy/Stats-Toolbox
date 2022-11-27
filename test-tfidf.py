import TF_tools as tfidf
import pandas as pd
df = pd.read_csv("../Amazon-data.csv")
sentences = df['reviews.text']
x_sent = []
for s in sentences:
    x_sent.append(s)
x_sent = x_sent[500:1000]
x = x_sent[20:50]

x = (tfidf.remove_stopwords(tfidf.lowercase(tfidf.remove_punctuation(x))))
words_set = tfidf.get_word_set(x, 'set')

df_tf = tfidf.TF_matrix(x, words_set, 'list')
df_tf_idf = tfidf.TF_IDF(x, words_set, df_tf, 'list')

df_tf_idf = pd.DataFrame(df_tf_idf)
print(df_tf_idf)
