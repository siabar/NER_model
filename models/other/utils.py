from pathlib import Path
import re
import torch
import os
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_data(file_path, max_length):
    subword_len_counter = 0
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    max_length -= tokenizer.num_special_tokens_to_add()
    token_docs = []
    label_docs = []

    with open(file_path, "rt") as f_p:
        tokens = []
        labels = []

        for line in f_p:
            line = line.rstrip()

            if not line:
                token_docs, label_docs, tokens, labels = add_to_list(token_docs, label_docs, tokens, labels)
                subword_len_counter = 0

                # if len(token_docs) == 100:
                #     break
                continue
            token, label = line.split("\t")

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            # I did not use truncation option on tokenizer, and keep extra length of input as a new sample.
            if (subword_len_counter + current_subwords_len) > max_length:
                token_docs, label_docs, tokens, labels = add_to_list(token_docs, label_docs, tokens, labels)
                tokens.append(token)
                labels.append(fixed_label(label))
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            tokens.append(token)
            labels.append(fixed_label(label))

    return token_docs, label_docs


def fixed_label(label):
    if label != "O":
        # add -disease to I and B tags
        return label + "-disease"
    else:
        return label


def add_to_list(token_docs, label_docs, tokens, labels):
    token_docs.append(tokens)
    label_docs.append(labels)
    tokens = []
    labels = []
    return token_docs, label_docs, tokens, labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
