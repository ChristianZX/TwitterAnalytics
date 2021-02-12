from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora, models, similarities
import pyLDAvis.gensim
from gensim.parsing.preprocessing import preprocess_documents
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import db_functions
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#from sklearn.feature_extraction.text import CountVectorizer


def topic_model_wordcloud(sql_raw):
    plt.figure()
    stance = ['links','rechts']
    for stance_index, stance_element in enumerate (stance):
        sql = sql_raw.replace('STANCE_REPLACE',stance_element)
        df = db_functions.select_from_db(sql)
        df.dropna(inplace=True)

        # vectorizer = CountVectorizer()
        # X = vectorizer.fit_transform(df['tweet'].tolist())
        # vectorizer.get_feature_names()

        tokens_input = df['tweet'].tolist()
        tokens_output = []
        for element in tokens_input:
            element = element.replace('"',"")
            element = element.replace('#', "")
            tokens_output.append(element.split())
        tokens = tokens_output
        #tokens = preprocess_documents(df['tweet'].tolist())
        stop_words = set(stopwords.words('german'))
        [x.upper() for x in stop_words]
        [x.title() for x in stop_words]

        stop_words.add('http')
        stop_words.add('Der')
        stop_words.add('Die')
        stop_words.add('Das')
        stop_words.add('mal')
        stop_words.add('Was')
        stop_words.add('Und')
        stop_words.add('Wir')


        #Stopword removal
        for index, element in enumerate(tokens):
            filtered_tweet = []
            for word in element:
                if word not in stop_words:
                    filtered_tweet.append(word)
                tokens[index] = filtered_tweet

        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]

        tfidf = models.TfidfModel(corpus)
        transformed_tfidf = tfidf[corpus]
        LDA_model = models.LdaMulticore(transformed_tfidf, num_topics=1, id2word=dictionary)


        plt.subplot(1, 2, stance_index+1)
        plt.title(stance_element)
        for t in range(LDA_model.num_topics):
            # fig, ax = plt.subplots()
            # ax.plot(x, y)
            # ax.set_title('A single plot')
            plt.imshow(WordCloud(random_state=42, min_word_length=3).fit_words(dict(LDA_model.show_topic(t, 20))))
            #plt.axis("off")
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(LDA_model, corpus, dictionary)
    # pyLDAvis.show(vis)
    plt.show()

if __name__ == '__main__':
    pass
