import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from bert_predict import BertPredict
from templates import female_groups, templates_female_map


remove = [
    'aneksi', 'anemone',  'blinci', 'bliski',  'bogki', 'boleni', 'bradni', 'braki',
    'bralci', 'delane', 'duhovi', 'eksota', 'eksote', 'eksoti',  'glasovali', 'hlajeni',
    'hlačni', 'homogene', 'homogeni', 'ideje', 'izolirni', 'izsevali', 'komiti', 'moževale',
    'muslini', 'nasilneževi', 'nasmehi', 'nasmeški',  'neveni', 'oble', 'obleteli', 'outi',
    'outputi', 'patricijski', 'permski', 'pernati', 'peroni', 'plevni', 'pobočni', 'politi',
    'premo', 'premočni', 'previle', 'prečne', 'prečni', 'reveževi', 'spremenljivka',
    'spremenljivke', 'spremni', 'svilni', 'temperaturne', 'univerza', 'univerze', 'veristi',
    'verižni', 'veronski', 'vzorčni', 'zloti', 'zložene', 'zloženi', 'zložni', 'zmesni',
    'šoki', "prejci", "prelci", "šarci", "eksplicirani", "šargi",
]

already_complete = [
    "Stari", "Stare", "Zdravi", "fit", "mojstri", "Romi", "Hrvati", "kralji",
    "Rusi", "prijatelji", "mami", "žene", "eni",  "sestra", "sestre", "mega",
    "tabu", "breme", "izziv", "zgled", "lepe", "lepi", "lepa", "fun", "top",
]


def remove_noninformative_words(df, wordforms_dict):
    df = df[~df.atribute.isin(wordforms_dict["conjunction"])]
    df = df[~df.atribute.isin(wordforms_dict["preposition"])]
    df = df[~df.atribute.isin(wordforms_dict["particle"])]
    df = df[~df.atribute.isin(wordforms_dict["pronoun"])]
    return df


def sloleks_to_dict(sloleks_path):
    """
    Explanation of MSD (morphosyntactic descriptions): https://nl.ijs.si/jos/msd/html-en/
    """
    wordforms_dict = {}
    sloleks = pd.read_table(
        sloleks_path,
        delim_whitespace=True,
        comment='*',
        names=["word", "lemma", "msd", "frequency"]
    )
    sloleks = sloleks[~sloleks.word.isin(remove)]
    sloleks["vrsta"] = sloleks["msd"].str[0]

    sloleks["število"] = np.where(sloleks['vrsta'] == 'P', sloleks["msd"].str[4], "")
    sloleks["število"] = np.where(sloleks['vrsta'] == 'S', sloleks["msd"].str[3], sloleks["število"])
    sloleks["število"] = np.where(sloleks['vrsta'] == 'G', sloleks["msd"].str[5], sloleks["število"])

    sloleks["spol"] = np.where(sloleks['vrsta'] == 'P', sloleks["msd"].str[3], "")
    sloleks["spol"] = np.where(sloleks['vrsta'] == 'S', sloleks["msd"].str[2], sloleks["spol"])
    sloleks["spol"] = np.where(sloleks['vrsta'] == 'G', sloleks["msd"].str[6], sloleks["spol"])

    sloleks["sklon"] = np.where(sloleks['vrsta'] == 'P', sloleks["msd"].str[5], "")
    sloleks["sklon"] = np.where(sloleks['vrsta'] == 'S', sloleks["msd"].str[4], sloleks["sklon"])

    # pridevniki moškega spola množine
    wordforms_dict["Pmmi"] = sloleks[
        (sloleks.vrsta == "P") & (sloleks.število == "m")
        & (sloleks.spol == "m") & (sloleks.sklon == "i")
    ].word.values

    # Pridevniki ženskega spola množine
    wordforms_dict["Pmzi"] = sloleks[
        (sloleks.vrsta == "P") & (sloleks.število == "m")
        & (sloleks.spol == "z") & (sloleks.sklon == "i")
    ].word.values

    wordforms_dict["Pmi"] = sloleks[
        (sloleks.vrsta == "P")
        & (sloleks.število == "m")
        & (sloleks.sklon == "i")
    ].word.values

    wordforms_dict["Smmi"] = sloleks[
        (sloleks.vrsta == "S") & (sloleks.število == "m")
        & (sloleks.spol == "m") & (sloleks.sklon == "i")
    ].word.values

    wordforms_dict["Smzi"] = sloleks[
        (sloleks.vrsta == "S") & (sloleks.število == "m")
        & (sloleks.spol == "z") & (sloleks.sklon == "i")
    ].word.values

    wordforms_dict["Smi"] = sloleks[
        (sloleks.vrsta == "S")
        & (sloleks.število == "m")
        & (sloleks.sklon == "i")
    ].word.values

    wordforms_dict["Sei"] = sloleks[
        (sloleks.vrsta == "S")
        & (sloleks.število == "e")
        & (sloleks.sklon == "i")
    ].word.values

    wordforms_dict["Sed"] = sloleks[
        (sloleks.vrsta == "S")
        & (sloleks.število == "e")
        & (sloleks.sklon == "d")
    ].word.values

    wordforms_dict["Gmm"] = sloleks[
        (sloleks.vrsta == "G")
        & (sloleks.število == "m")
        & (sloleks.spol == "m")
    ].word.values

    wordforms_dict["Gmz"] = sloleks[
        (sloleks.vrsta == "G")
        & (sloleks.število == "m")
        & (sloleks.spol == "z")
    ].word.values

    wordforms_dict["Gm"] = sloleks[
        (sloleks.vrsta == "G") & (sloleks.število == "m")
    ].word.values

    wordforms_dict["adverb"] = sloleks[sloleks.vrsta == "R"].word.values
    wordforms_dict["conjunction"] = sloleks[sloleks.vrsta == "V"].word.values
    wordforms_dict["particle"] = sloleks[sloleks.vrsta == "L"].word.values
    wordforms_dict["preposition"] = sloleks[sloleks.vrsta == "D"].word.values
    wordforms_dict["interjection"] = sloleks[sloleks.vrsta == "M"].word.values
    wordforms_dict["pronoun"] = sloleks[sloleks.vrsta == "Z"].word.values

    return wordforms_dict


def complete_subwords(df, model, continuations, wordforms_dict):

    def complete(row):
        possible_words = []
        atr = row["atribute"]
        template = row["template"]
        target = row["target"]

        if atr in already_complete:
            return [atr, row["prob"], row["prior"], row["score"], ""]

        if target in female_groups:
            template = templates_female_map[template]
            if "tako" in template or "zelo" in template:
                if (
                    atr not in wordforms_dict["Pmzi"]
                    and atr not in wordforms_dict["Gmz"]
                    and atr not in wordforms_dict["adverb"]
                ):
                    possible_words = (
                        continuations[("Pmzi", atr)]
                        + continuations[("Gmz", atr)]
                    )

            elif template == "[TARGET] so obravnavane kot [MASK].":
                if (
                    atr not in wordforms_dict["Pmzi"]
                    and atr not in wordforms_dict["Smi"]
                    and atr not in wordforms_dict["Sei"]
                    and atr not in wordforms_dict["adverb"]
                ):
                    possible_words = (
                        continuations[("Pmzi", atr)]
                        + continuations[("Smzi", atr)]
                        + continuations[("Sei", atr)]
                    )
            else:
                if (
                    atr not in wordforms_dict["Pmzi"]
                    and atr not in wordforms_dict["Smi"]
                    and atr not in wordforms_dict["Gmz"]
                    and atr not in wordforms_dict["adverb"]
                ):
                    possible_words = (
                        continuations[("Pmzi", atr)]
                        + continuations[("Smzi", atr)]
                        + continuations[("Gmz", atr)]
                    )

        else:  # male groups
            if "tako" in template or "zelo" in template:
                if (
                    atr not in wordforms_dict["Pmmi"]
                    and atr not in wordforms_dict["Gmm"]
                    and atr not in wordforms_dict["adverb"]
                ):
                    possible_words = (
                        continuations[("Pmmi", atr)]
                        + continuations[("Gmm", atr)]
                    )

            elif template == "[TARGET] so obravnavani kot [MASK].":
                if (
                    atr not in wordforms_dict["Pmmi"]
                    and atr not in wordforms_dict["Smi"]
                    and atr not in wordforms_dict["Sei"]
                    and atr not in wordforms_dict["adverb"]
                ):
                    possible_words = (
                        continuations[("Pmmi", atr)]
                        + continuations[("Smmi", atr)]
                        + continuations[("Sei", atr)]
                    )
            else:
                if (
                    atr not in wordforms_dict["Pmmi"]
                    and atr not in wordforms_dict["Smi"]
                    and atr not in wordforms_dict["Gmm"]
                    and atr not in wordforms_dict["adverb"]
                ):
                    possible_words = (
                        continuations[("Pmmi", atr)]
                        + continuations[("Smmi", atr)]
                        + continuations[("Gmm", atr)]
                    )

        if possible_words:
            best_word, new_prob, new_prior, new_score = model.select_best_word(
                target, template, possible_words
            )
            return [best_word, new_prob, new_prior, new_score, atr]
        else:
            # return original values
            return [atr, row["prob"], row["prior"], row["score"], ""]

    tqdm.pandas()
    df[["atribute", "prob", "prior", "score", "atr_prev"]] = df.progress_apply(complete, axis=1, result_type="expand")

    return df


def precompute_word_continuations(df, wordforms_dict):
    continuations = {}
    atrs = df.atribute.unique()
    wordforms = ["Pmmi", "Smmi", "Pmzi", "Smmi", "Smzi", "Sei", "Gmm", "Gmz"]
    for wordform in wordforms:
        for atr in atrs:
            continuations[(wordform, atr)] = [
                w for w in wordforms_dict[wordform] if w.startswith(atr)
            ]
    return continuations


if __name__ == "__main__":

    model_name = "sloberta"
    sloleks_path = "lexicons/Sloleks2.0.MTE/sloleks_clarin_2.0-sl.tbl"
    input_df_path = f"{model_name}/top_atributes.csv"
    output_df_path = f"{model_name}/top_atributes_completed.csv"

    bert_model = BertPredict(model_name)

    wordforms_dict = sloleks_to_dict(sloleks_path)

    df_all = pd.read_csv(input_df_path)

    print(len(df_all))

    df = remove_noninformative_words(df_all, wordforms_dict)
    print(len(df), len(df)/len(df_all))

    continuations = precompute_word_continuations(df, wordforms_dict)

    dfc = complete_subwords(df, bert_model, continuations, wordforms_dict)
    dfc.to_csv(output_df_path, index=False)
