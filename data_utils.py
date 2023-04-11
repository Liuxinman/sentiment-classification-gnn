# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from transformers import AutoTokenizer


def convert_tokens_to_flags(lst, flag=2):
    word_count = 0
    for i in range(0, len(lst)):
        if not lst[i].startswith("##"):
            word_count += 1
    return [flag] * word_count


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, "r", encoding="utf-8", newline="\n", errors="ignore")
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = " ".join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype="float32")
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    # embedding_matrix_file_name = "{0}_{1}_embedding_matrix.pkl".format(str(embed_dim), type)
    embedding_matrix_file_name = "/content/sentiment-classification-gnn/300_rest14_embedding_matrix.pkl"
    # if os.path.exists(embedding_matrix_file_name):
    if True:
        print("loading embedding_matrix:", embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, "rb"))
    else:
        print("loading word vectors ...")
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(
            -1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim)
        )
        fname = "./glove/glove.840B.300d.txt"
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print("building embedding_matrix:", embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, "wb"))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx["<pad>"] = self.idx
            self.idx2word[self.idx] = "<pad>"
            self.idx += 1
            self.word2idx["<unk>"] = self.idx
            self.idx2word[self.idx] = "<unk>"
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ""
        for fname in fnames:
            fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer=None, use_bert=False):
        fin = open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
        lines = fin.readlines()
        fin.close()
        fin = open(fname + ".tree", "rb")
        idx2tree = pickle.load(fin)
        fin.close()
        if use_bert:
            fname += ".bert"
        fin = open(fname + ".graph", "rb")
        idx2graph = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_type_ids = None
            text_attention_mask = None
            text_word_ids = None
            text_indices_bert = None
            if use_bert:
                text_indices = convert_tokens_to_flags(
                    tokenizer.tokenize(text_left + " " + aspect + " " + text_right)
                )
                context_indices = convert_tokens_to_flags(
                    tokenizer.tokenize(text_left + " " + text_right)
                )
                aspect_indices = convert_tokens_to_flags(tokenizer.tokenize(aspect))
                left_indices = convert_tokens_to_flags(tokenizer.tokenize(text_left))
                text_indices_bert = tokenizer(text_left + " " + aspect + " " + text_right, aspect)
                text_word_ids = text_indices_bert.word_ids()
                for j in range(len(text_word_ids)):
                    if text_word_ids[j] == None:
                        text_word_ids[j] = -1
                text_type_ids = text_indices_bert["token_type_ids"]
                text_attention_mask = text_indices_bert["attention_mask"]
                text_indices_bert = text_indices_bert["input_ids"]
            else:
                text_indices = tokenizer.text_to_sequence(
                    text_left + " " + aspect + " " + text_right
                )
                context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                left_indices = tokenizer.text_to_sequence(text_left)

            polarity = int(polarity) + 1
            dependency_graph = idx2graph[i]
            dependency_tree = idx2tree[i]

            data = {
                "text_indices": text_indices,
                "context_indices": context_indices,
                "aspect_indices": aspect_indices,
                "left_indices": left_indices,
                "polarity": polarity,
                "dependency_graph": dependency_graph,
                "dependency_tree": dependency_tree,
                "text_type_ids": text_type_ids,
                "text_attention_mask": text_attention_mask,
                "text_word_ids": text_word_ids,
                "text_indices_bert": text_indices_bert,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset="twitter", embed_dim=300, use_bert=False, bert_version="prajjwal1/bert-small"):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            "twitter": {
                "train": "./datasets/acl-14-short-data/train.raw",
                "test": "./datasets/acl-14-short-data/test.raw",
            },
            "rest14": {
                "train": "/content/sentiment-classification-gnn/datasets/semeval14/restaurant_train.raw",
                "test": "/content/sentiment-classification-gnn/datasets/semeval14/restaurant_test.raw",
            },
            "lap14": {
                "train": "./datasets/semeval14/laptop_train.raw",
                "test": "./datasets/semeval14/laptop_test.raw",
            },
            "rest15": {
                "train": "./datasets/semeval15/restaurant_train.raw",
                "test": "./datasets/semeval15/restaurant_test.raw",
            },
            "rest16": {
                "train": "./datasets/semeval16/restaurant_train.raw",
                "test": "./datasets/semeval16/restaurant_test.raw",
            },
        }
        self.embedding_matrix = None
        if use_bert:
            tokenizer = AutoTokenizer.from_pretrained(bert_version)
            self.train_data = ABSADataset(
                ABSADatesetReader.__read_data__(fname[dataset]["train"], tokenizer, use_bert=True)
            )
            self.test_data = ABSADataset(
                ABSADatesetReader.__read_data__(fname[dataset]["test"], tokenizer, use_bert=True)
            )
        else:
            if os.path.exists(dataset + "_word2idx.pkl"):
                print("loading {0} tokenizer...".format(dataset))
                with open(dataset + "_word2idx.pkl", "rb") as f:
                    word2idx = pickle.load(f)
                    tokenizer = Tokenizer(word2idx=word2idx)
            else:
                text = ABSADatesetReader.__read_text__(
                    [fname[dataset]["train"], fname[dataset]["test"]]
                )
                tokenizer = Tokenizer()
                tokenizer.fit_on_text(text)
                with open(dataset + "_word2idx.pkl", "wb") as f:
                    pickle.dump(tokenizer.word2idx, f)
            self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
            self.train_data = ABSADataset(
                ABSADatesetReader.__read_data__(fname[dataset]["train"], tokenizer)
            )
            self.test_data = ABSADataset(
                ABSADatesetReader.__read_data__(fname[dataset]["test"], tokenizer)
            )
