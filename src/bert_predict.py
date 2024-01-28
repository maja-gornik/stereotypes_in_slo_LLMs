import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from helpers import clean_data
from templates import templates, groups_dict, female_groups, templates_female_map

priors = {}


class BertPredict:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = self.get_bert_and_tokenizer(self.model_name)

    @staticmethod
    def get_bert_and_tokenizer(model_name):
        if model_name == "sloberta":
            tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", use_fast=False)
            model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/sloberta")
        elif model_name == "crosloengual-bert":
            tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert", use_fast=False)
            model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/crosloengual-bert")
        elif model_name == "sleng-bert":
            tokenizer = AutoTokenizer.from_pretrained("cjvt/sleng-bert")
            model = AutoModelForMaskedLM.from_pretrained("cjvt/sleng-bert")
        elif model_name == "sloberta-sleng":
            tokenizer = AutoTokenizer.from_pretrained("cjvt/sloberta-sleng")
            model = AutoModelForMaskedLM.from_pretrained("cjvt/sloberta-sleng")
        elif model_name == "roberta-base":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Model {model_name} not supported!")
        return model, tokenizer

    def fill_mask(self, sentence, n=200):
        """
        Returns most probable tokens and their probabilities in a masked sentence.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(**inputs).logits

        mask_token_logits = token_logits[0, mask_token_index[-1], :]
        top_token_ids = torch.topk(mask_token_logits, n).indices.tolist()

        sm = torch.nn.Softmax(dim=0)
        probs = sm(mask_token_logits)
        return top_token_ids, probs.detach().numpy()

    def get_top_atributes(self, target, template, n=200):
        """
        Returns atributes that are most tipical for the target in the given sentence
        """
        sentence = template.replace("[MASK]", self.tokenizer.mask_token)

        sent1 = sentence.replace("[TARGET]", target)
        top_token_ids, probs = self.fill_mask(sent1, n=n)

        sent2 = sentence.replace("[TARGET]", self.tokenizer.mask_token)
        _, probs2 = self.fill_mask(sent2, n=n)

        atributes = self.tokenizer.batch_decode(top_token_ids)
        atr_tokens = self.tokenizer.convert_ids_to_tokens(top_token_ids)

        df = pd.DataFrame({
            "atribute": atributes,
            "token": atr_tokens,
            "prob": probs[top_token_ids],
            "prior": probs2[top_token_ids]
        })
        df["score"] = np.log(df["prob"]/df["prior"])
        return df

    def whole_word_prob_batch(self, target, sentence, atributes, mask_target=False):
        """
        Calculate probability of atributes in a given sentence 

        atributes can consist of multiple tokens -> we replace each token with
        one <MASK> and multiply probablities for each mask position
        """
        atribute_tokens = {}
        senetences = []
        max_num_tokens = 0
        for atr in atributes:
            # how many (mask) tokens do we need
            # remove start and end token
            atr_tokens = self.tokenizer(atr)["input_ids"][1:-1]
            atribute_tokens[atr] = atr_tokens
            max_num_tokens = max(max_num_tokens, len(atr_tokens))

        target = self.tokenizer.mask_token if mask_target else target
        for num_tokens in range(0, max_num_tokens):
            s = sentence.replace("[MASK]", self.tokenizer.mask_token*(num_tokens+1))
            s = s.replace("[TARGET]", target)
            senetences.append(s)

        inputs = self.tokenizer(senetences, return_tensors="pt", padding="longest")
        token_logits = self.model(**inputs).logits

        sm = torch.nn.Softmax(dim=2)
        token_probs = sm(token_logits)

        atr_probs = {}
        for atr in atributes:
            atr_tokens = atribute_tokens[atr]
            ind = len(atr_tokens)-1
            mask_idx = torch.where(inputs["input_ids"][ind] == self.tokenizer.mask_token_id)[0]
            if mask_target or target == "<mask>":
                mask_idx = mask_idx[1:]  # skip first mask, it represents a target

            p_word2 = token_probs[ind, mask_idx, atr_tokens].prod().item()
            atr_probs[atr] = p_word2

        return atr_probs

    def select_best_word(self, target, sentence, atributes):
        """select the word from the list of atributes that has the highest
        probability in the given sentence"""

        probs_dict = self.whole_word_prob_batch(
            target, sentence, atributes, mask_target=False
        )
        best_word = max(probs_dict, key=probs_dict.get)
        max_prob = probs_dict[best_word]

        if (sentence, best_word) in priors:
            prior = priors[(sentence, best_word)]
        else:
            p_dict = self.whole_word_prob_batch(
                target, sentence, [best_word], mask_target=True
            )
            prior = p_dict[best_word]
            priors[(sentence, best_word)] = prior

        score = np.log(max_prob/prior)

        return best_word, max_prob, prior, score


if __name__ == "__main__":

    model_name = "sloberta"  # sloberta-sleng

    bert = BertPredict(model_name)
    # res = bert.get_top_atributes("ženske", "Zakaj so [TARGET] tako [MASK]?")

    results = []
    for group in groups_dict.keys():
        for target in tqdm(groups_dict[group]):
            for template in templates[:5]:

                if target in female_groups:
                    t = templates_female_map[template]
                else:
                    t = template

                df = bert.get_top_atributes(target, t)
                df["target"] = target
                df["template"] = template
                df["group"] = group
                df = df.sort_values(by="score", ascending=False)
                results.append(df)

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    df_all = pd.concat(results)
    df_all.to_csv(f"{model_name}/top_atributes_all.csv", index=False)

    df_clean = clean_data(df_all)
    df_clean.to_csv(f"{model_name}/top_atributes.csv", index=False)
