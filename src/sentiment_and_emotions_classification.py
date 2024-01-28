import pandas as pd
from lemmagen3 import Lemmatizer
from tqdm import tqdm


def get_emotions(df, slo_lexicon, eng_lexicon):
    """
    Obtain sentiment and emotions for each word using emotion lexicon
    """

    lem_sl = Lemmatizer('sl')

    emotions = [
        'Negative', 'Positive', 'Disgust', 'Anger', 'Sadness',
        'Fear', 'Trust', 'Joy', 'Surprise', 'Anticipation',
    ]

    not_found = 0
    not_found_list = []

    for word in tqdm(df.atribute.unique()):
        word = word.lower()

        lexicon_row = eng_lexicon.loc[
            eng_lexicon['English (en)'].str.lower() == word
        ]

        if lexicon_row.empty:
            lexicon_row = slo_lexicon.loc[slo_lexicon['SL'] == word]

        if lexicon_row.empty:
            word_l = lem_sl.lemmatize(word)
            lexicon_row = slo_lexicon.loc[slo_lexicon['SL'] == word_l]

        if lexicon_row.empty:
            not_found += 1
            not_found_list.append(word)
        else:
            for emotion in emotions:
                df.loc[df.atribute == word, emotion] = lexicon_row[emotion].values[0]

    print(not_found_list)
    print(f"Number of words not found: {not_found}")
    print(f"Number of all words: {len(df.atribute.unique())}")
    return df


if __name__ == "__main__":

    slo_lexicon_path = "lexicons/LiLaH-HR-NL-SL.tsv"
    additional_lexicon_path = "lexicons/emotions_added.csv"
    eng_lexicon_path = "lexicons/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx"

    model_name = "sloberta"
    input_df_path = f"{model_name}/top_atributes_completed.csv"
    result_df_path = f"{model_name}/atributes_emotions.csv"

    df = pd.read_csv(input_df_path)

    slo_lexicon_df = pd.read_csv(slo_lexicon_path, sep="\t")
    emotions_added = pd.read_csv(additional_lexicon_path)
    slo_lexicon_df = pd.concat([emotions_added, slo_lexicon_df], ignore_index=True)
    slo_lexicon_df['SL'] = slo_lexicon_df['SL'].str.lower()

    # some values consist of multiple words, separated by comma,
    # we put them into separate rows
    slo_lexicon_df['SL'] = slo_lexicon_df['SL'].str.split(", ")
    slo_lexicon_df = slo_lexicon_df.explode('SL').reset_index(drop=True)

    eng_lexicon_df = pd.read_excel(eng_lexicon_path)

    df_with_emotions = get_emotions(df, slo_lexicon_df, eng_lexicon_df)

    df_with_emotions.to_csv(result_df_path, index=False)
