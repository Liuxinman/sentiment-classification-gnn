# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key="text_indices", shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size : (i + 1) * batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        # batch_dependency_tree = []
        batch_text_type_ids = []
        batch_text_attention_mask = []
        batch_text_indices_bert = []
        batch_word_ids = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        if batch_data[0]["text_indices_bert"] != None:
            max_len_bert = max([len(t["text_indices_bert"]) for t in batch_data])
        for item in batch_data:
            (
                text_indices,
                context_indices,
                aspect_indices,
                left_indices,
                polarity,
                dependency_graph,
                dependency_tree,
                text_type_ids,
                text_attention_mask,
                text_word_ids,
                text_indices_bert,
            ) = (
                item["text_indices"],
                item["context_indices"],
                item["aspect_indices"],
                item["left_indices"],
                item["polarity"],
                item["dependency_graph"],
                item["dependency_tree"],
                item["text_type_ids"],
                item["text_attention_mask"],
                item["text_word_ids"],
                item["text_indices_bert"],
            )
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(
                numpy.pad(
                    dependency_graph,
                    ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                    "constant",
                )
            )
            # batch_dependency_tree.append(
            #     numpy.pad(
            #         dependency_tree,
            #         ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
            #         "constant",
            #     )
            # )

            # BERT
            if text_indices_bert != None:
                text_indices_bert_padding = [0] * (max_len_bert - len(text_indices_bert))
                text_type_ids_padding = [0] * (max_len_bert - len(text_type_ids))
                text_attention_mask_padding = [0] * (max_len_bert - len(text_attention_mask))
                text_word_ids_padding = [-1] * (max_len_bert - len(text_word_ids))
                batch_text_type_ids.append(text_type_ids + text_type_ids_padding)
                batch_text_attention_mask.append(text_attention_mask + text_attention_mask_padding)
                batch_text_indices_bert.append(text_indices_bert + text_indices_bert_padding)
                batch_word_ids.append(text_word_ids + text_word_ids_padding)

        return {
            "text_indices": torch.tensor(batch_text_indices),
            "context_indices": torch.tensor(batch_context_indices),
            "aspect_indices": torch.tensor(batch_aspect_indices),
            "left_indices": torch.tensor(batch_left_indices),
            "polarity": torch.tensor(batch_polarity),
            "dependency_graph": torch.tensor(batch_dependency_graph),
            # "dependency_tree": torch.tensor(batch_dependency_tree),
            "text_type_ids": torch.tensor(batch_text_type_ids),
            "text_attention_mask": torch.tensor(batch_text_attention_mask),
            "text_indices_bert": torch.tensor(batch_text_indices_bert),
            "text_word_ids": torch.tensor(batch_word_ids),
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
