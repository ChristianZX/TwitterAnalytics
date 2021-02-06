#######################
###WORK IN PROGRESS!###
#######################

from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora, models, similarities
import pyLDAvis.gensim
from gensim.parsing.preprocessing import preprocess_documents
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import db_functions
import matplotlib.pyplot as plt
from wordcloud import WordCloud

if __name__ == '__main__':

    sql_right ="""
    select tweet from v_all_hashtags
    where from_staging_table like '%150%'
    and combined_rating = 'right wing'
    """

    # sql_left = """
    # select tweet from v_all_hashtags
    # where from_staging_table like '%150%'
    # and combined_rating = 'moderate'
    # """

    #left
    sql = """
    select distinct h.tweet from s_h_150jahrevaterland_20210204_2312 h, n_users u 
    where h.user_id = u.id
    and u.combined_rating = 'rechts'
    and combined_conf >= 0.7
    """

    # sql_combined = """
    # select tweet from v_all_hashtags
    # where 1=1
    # --and from_staging_table like '%150%'
    # and combined_rating in ('right wing','moderate')
    # --limit 1000
    # """

    df = db_functions.select_from_db(sql)

    # print unprocessed text
    #print(df.tweet[0])
    # print processed text
    #print(preprocess_string(df.tweet[0]))

    df.dropna(inplace=True)

    tokens = preprocess_documents(df['tweet'].tolist())

    #tokens = preprocess_string(df['tweet'].tolist())
    #tokens = (preprocess_string(df.tweet[0]))

    # german_stop_words = stopwords.words('german')
    # vect = CountVectorizer(stop_words=german_stop_words)  # Now use this in your pipeline
    #
    #stop_words = set(stopwords.words('english'))
    stop_words = set(stopwords.words('german'))
    stop_words.add('http')
    #word_tokens = word_tokenize(example_sent)

    #[w for w in tokens[0] if not w in stop_words]

    #filtered_sentence = [w for w in tokens if not w in stop_words]
    # filtered_sentence = []
    # for w in tokens:
    #     if w not in stop_words:
    #         filtered_sentence.append(w)


    for index, element in enumerate (tokens):
        filtered_tweet = []
        for word in element:
            if word not in stop_words:
                filtered_tweet.append(word)
            tokens[index] = filtered_tweet


    dictionary = corpora.Dictionary(tokens)
    #print (dictionary)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    #Allocation of word to frequency.
    #[[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]

    tfidf = models.TfidfModel(corpus)
    transformed_tfidf = tfidf[corpus]
    LDA_model = models.LdaMulticore(transformed_tfidf, num_topics=1, id2word=dictionary)
    #LDA_model = models.ldamodel(transformed_tfidf, num_topics=2)
    #print(LDA_model.show_topics())


    for t in range(LDA_model.num_topics):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(LDA_model.show_topic(t, 20))))
        plt.axis("off")
        #plt.title("Topic #" + str(t))
        plt.show()

    #pyLDAvis.enable_notebook()
    #vis = pyLDAvis.gensim.prepare(LDA_model, corpus, dictionary)
    #pyLDAvis.show(vis)


    print ("Bing")
