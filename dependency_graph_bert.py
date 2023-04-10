import numpy as np
import spacy
import pickle

from spacy.tokens import Doc
from transformers import BertTokenizer


def convert_subwords_to_words(lst):
    out = []
    stack = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i].startswith("##"):
            stack.append(lst[i][2:])
        else:
            out.append("".join(stack))
            stack = [lst[i]]
    if len(stack) >= 1:
        out.append("".join(stack))
    return out


class BertWordLevelTokenizer(object):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, text):
        words = self.tokenizer.tokenize(text)
        words = convert_subwords_to_words(words)
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = BertWordLevelTokenizer(
    nlp.vocab, BertTokenizer.from_pretrained("bert-base-uncased")
)


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = nlp(text)
    words = convert_subwords_to_words(tokenizer.tokenize(text))
    matrix = np.zeros((len(words), len(words))).astype("float32")
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


def process(filename):
    fin = open(filename, "r", encoding="utf-8", newline="\n", errors="ignore")
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + ".bert.graph", "wb")
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left + " " + aspect + " " + text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()


if __name__ == "__main__":
    process("./datasets/semeval14/restaurant_train.raw")
    process("./datasets/semeval14/restaurant_test.raw")
