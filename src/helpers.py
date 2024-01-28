import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


def generate_wordcloud(df, group):
    """
    wordcloud of atributes for specified group, colorcoded by sentiment
    """

    def color_func(word, *args, **kwargs):
        if df[df.atribute == word].Positive.values[0] == 1:
            color = '#1f77b4'  # blue-positive
        elif df[df.atribute == word].Negative.values[0] == 1:
            color = '#d62728'  # red-negative
        else:
            color = '#000000'  # black-neutral
        return color

    atributes = " ".join(df[df.target == group].atribute.tolist())

    wordcloud = WordCloud(
        background_color='white',
        collocations=False,
        max_font_size=150,
        min_font_size=10,
        color_func=color_func,
        min_word_length=3,
        width=1000,
        height=500,
    ).generate(atributes)

    return wordcloud


def generate_wordcloud_tfidf(df, df_tfidf, group):
    """
    wordcloud of atributes for specified group, colorcoded by sentiment
    """

    def color_func(word, *args, **kwargs):
        if df[df.atribute_l == word].Positive.values[0] == 1:
            color = '#1f77b4'  # blue-positive
        elif df[df.atribute_l == word].Negative.values[0] == 1:
            color = '#d62728'  # red-negative
        else:
            color = '#000000'  # black-neutral
        return color

    atributes = df[df.target == group].atribute_l.tolist()

    atribute_scores = {}
    df_g = df_tfidf[df_tfidf.target == group]
    for a in atributes:
        atribute_scores[a] = df_g[df_g.atribute_l == a].tf_idf.values[0]

    wordcloud = WordCloud(
        background_color='white',
        collocations=False,
        max_font_size=150,
        min_font_size=10,
        color_func=color_func,
        min_word_length=3,
        width=1000,
        height=500,
    ).generate_from_frequencies(atribute_scores)

    return wordcloud


def clean_data(df):
    df = df[df["atribute"].notna()]
    df["atribute"] = df["atribute"].str.strip()
    df = df[df["atribute"].str.len() > 2]
    df = df[df["atribute"].str.isalpha()]
    # only tokens that reperesent start of the word
    df = df[df.token.str.startswith("▁")]
    return df


def jaccard(a, b):
    i = set(a).intersection(set(b))
    u = set(a).union(set(b))
    return len(i) / len(u)


def get_tf_idf_scores(df):
    """
    TF-IDF-like scoring of atribute importance
    """
    targets = df.target.unique()

    # create corpus, one document contains all atributes for one target
    corpus = []
    for t in targets:
        df_t = df[df.target == t]
        document = " ".join(df_t.atribute_l.values)
        corpus.append(document)

    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(corpus)  # group x atribute
    feature_names = vectorizer.get_feature_names_out()

    # in how many documents(diferent targets) atribute appears
    count_doc = (counts != 0).sum(0)
    idf = np.log(len(corpus) / count_doc) + 1
    idf_list = np.array(idf)[0]
    df_idf = pd.DataFrame({"atribute_l": feature_names, "idf": idf_list})

    df_tf = df.groupby(["target", "atribute_l"]).score.sum(
    ).reset_index()  # sum scores per target and atribute
    df_num = df.groupby("target").size().reset_index(
        name="n")  # number of atributes per target
    df_tfidf = df_tf.merge(df_num, on="target", how="left").merge(
        df_idf, on="atribute_l", how="left")
    df_tfidf["tf_idf"] = df_tfidf["score"] / df_tfidf["n"] * df_tfidf["idf"]
    return df_tfidf
