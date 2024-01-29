from extractors.base import BaseExtractor

import os
import torch
import numpy as np
import torch.nn as nn
from logger import logger
from os import path
from typing import List, Tuple, Dict, Any

from torch_scatter import scatter_mean
from tqdm import tqdm
from conllu import parse_incr, TokenList
from os import path
import yaml
from transformers import BertTokenizer, BertModel
import pickle
from argparse import ArgumentParser
from ud_treebank_utils.embedders import FastTextEmbedder
from ud_treebank_utils.parser import parse_unimorph_features
from config import cfg as default_cfg

class IntrinsicProbingExtractor(nn.Module, BaseExtractor):
    def __init__(self, cfg, model, tokenizer,
                     is_hf_model=True, is_fasttext=False, limit_number=None, skip_existing=False):
        super().__init__()
        self.cfg      = cfg
        self.device   = self.cfg['device']
        self.is_hf_model = is_hf_model
        self.is_fasttext = is_fasttext
        self.model       = model
        self.tokenizer   = tokenizer
        self.limit_number = limit_number
        self.skip_existing = skip_existing
        if not (self.is_hf_model or self.is_fasttext) or (self.is_hf_model and self.is_fasttext):
            raise Exception("Must do either FastText or HF MODEL, but not both")
    
    @classmethod
    def code(cls):
        return 'intr_prob'
    
    def activation_func(self, x, concept_idx):
        NotImplemented
    
    def extract_concepts(self, language):
        treebank_path = os.path.join(self.cfg['data_dir'], 'ud/ud-treebanks-v2.1/' + language)
        for f in os.listdir(treebank_path):
            if path.isfile(path.join(treebank_path, f)) and "-um-" in f and f.endswith(".conllu"):
                filename = f
                full_path = path.join(treebank_path, filename)
                total = 0
                skipped: Dict[str, int] = {}

                # Load possible UM tags
                tags_file = path.join(
                    self.cfg['data_dir'],
                    f"ud/ud-treebanks-v2.1/tags.yaml"
                ) #modified
                with open(tags_file, 'r') as h:
                    _UNIMORPH_ATTRIBUTE_VALUES = yaml.full_load(h)["categories"]

                _UNIMORPH_VALUES_ATTRIBUTE = {v: k for k, vs in _UNIMORPH_ATTRIBUTE_VALUES.items() for v in vs}

                # Setup UM feature parsing
                _FEATS = ["id", "form", "lemma", "upos", "xpos", "um_feats", "head", "deprel", "deps", "misc"]

                # Parse Conll-U files with UM
                final_token_list: List[TokenList] = []
                with open(full_path, "r") as h:
                    # Setup BERT tokenizer here provisionally as we need to know which sentences have over 512 subtokens
                    tokenizer = self.tokenizer

                    for sent_id, tokenlist in enumerate(tqdm(
                            parse_incr(h, fields=_FEATS, field_parsers={"um_feats": self.unimorph_feature_parser}))):
                        # Only process first `limit_number` if it is set
                        if self.limit_number is not None and sent_id > self.limit_number:
                            break

                        # Remove virtual nodes
                        tokenlist = [t for t in tokenlist if not (isinstance(t["id"], tuple) and t["id"][1] == ".")]

                        # Build list of ids that are contracted
                        contracted_ids: List[int] = []
                        for t in tokenlist:
                            if isinstance(t["id"], tuple):
                                if t["id"][1] == "-":
                                    # Range
                                    contracted_ids.extend(list(range(t["id"][0], t["id"][2] + 1)))
                                    # Build dictionary of non-contracted token ids to tokens
                        non_contracted_token_dict: Dict[int, Any] = {
                            t["id"]: t for t in tokenlist if not isinstance(t["id"], tuple)}

                        # Build final list of (real) tokens, without any contractions
                        # Contractions are assigned the attributes of the constituent words, unless there is a clash
                        # with one attribute taking more than one value (e.g. POS tag is a frequent example), whereby
                        # we discard it.
                        final_tokens: List[Any] = []
                        for t in tokenlist:
                            if isinstance(t["id"], tuple):
                                constituent_ids = list(range(t["id"][0], t["id"][2] + 1))
                                t["um_feats"] = self.merge_attributes(
                                    [non_contracted_token_dict[x] for x in constituent_ids],
                                    _UNIMORPH_VALUES_ATTRIBUTE)

                                # If this is a contraction, add it
                                final_tokens.append(t)
                            elif t["id"] not in contracted_ids:
                                # Check if this t is part of a contraction
                                final_tokens.append(t)

                        final_tokens: TokenList = TokenList(final_tokens)
                        # Skip if this would have more than 512 subtokens
                        labelled_subwords = self.subword_tokenize(tokenizer, [t["form"] for t in final_tokens])
                        subtoken_indices, subtokens = zip(*labelled_subwords)
                        if len(subtokens) >= 512:
                            if "subtoken_count" not in skipped:
                                skipped["subtoken_count"] = 0

                            skipped["subtoken_count"] += 1
                            continue

                        if "total_sents" not in skipped:
                            skipped["total_sents"] = 0

                        skipped["total_sents"] += 1

                        # Add this sentence to the list we are processing
                        final_token_list.append(final_tokens)
                        # Print logs:
                print("Skipped:")
                print(skipped)
                print()

                print(f"Total: {total}")
                final_results = []
                if self.is_hf_model:
                    # modified
                    output_filename = filename.split('.')[0] + "-{}.pkl".format("bert-base-multilingual-cased")
                    # print('output_filename:', output_filename)
                    output_file = path.join(treebank_path, output_filename)
                    # print('output_file:', output_file)

                    if self.skip_existing and path.exists(output_file):
                        print(f"Skipping {filename}. Reason: file already processed")
                        continue

                    print(f"Processing {filename}...")
                    # Setup BERT
                    model = self.model
                    print(f"loading model successfully!")
                    print()
                    results = []
                    for sent_id, tokenlist in enumerate(tqdm(final_token_list)):
                        labelled_subwords = self.subword_tokenize(tokenizer, [t["form"] for t in tokenlist])
                        subtoken_indices, subtokens = zip(*labelled_subwords)
                        subtoken_indices_tensor = torch.tensor(subtoken_indices).to(self.device)

                        # We add special tokens to the sequence and remove them after getting the BERT output
                        subtoken_ids = torch.tensor(
                            tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(subtokens))).to(self.device)

                        results.append((tokenlist, subtoken_ids, subtoken_indices_tensor))

                    # Prepare to compute BERT embeddings
                    model.eval()
                    # NOTE: No batching, right now. But could be worthwhile to implement if a speed-up is necessary.
                    for token_list, subtoken_ids, subtoken_indices_tensor in tqdm(results):
                        total += 1

                        with torch.no_grad():
                            # shape: (batch_size, max_seq_length_in_batch + 2)
                            inputs = subtoken_ids.reshape(1, -1)

                            # shape: (batch_size, max_seq_length_in_batch)
                            indices = subtoken_indices_tensor.reshape(1, -1)

                            # shape: (batch_size, max_seq_length_in_batch + 2, embedding_size)
                            outputs = model(inputs)
                            final_output = outputs[0]

                            # shape: (batch_size, max_seq_length_in_batch, embedding_size)
                            # Here we remove the special tokens (BOS, EOS)
                            final_output = final_output[:, 1:, :][:, :-1, :]

                            # Average subtokens corresponding to the same word
                            # shape: (batch_size, max_num_tokens_in_batch, embedding_size)
                            token_embeddings = scatter_mean(final_output, indices, dim=1)

                        # Convert to python objects
                        embedding_list = [x.cpu().numpy() for x in token_embeddings.squeeze(0).split(1, dim=0)]

                        for t, e in zip(token_list, embedding_list):
                            t["embedding"] = e

                        final_results.append(token_list)
                
                # Keep important parts
                final_results_filtered = []
                for row in final_results:
                    for token in row:
                        final_results_filtered.append({
                            "word": token["form"],
                            "embedding": token["embedding"],
                            "attributes": token["um_feats"],
                        })

                # Save final results
                with open(output_file, "wb") as h:
                    pickle.dump(final_results_filtered, h)


    def subword_tokenize(self, tokenizer: BertTokenizer, tokens: List[str]) -> List[Tuple[int, str]]:
        """
        Returns: List of subword tokens, List of indices mapping each subword token to one real token.
        """
        subtokens = [tokenizer.tokenize(t) for t in tokens]

        indexed_subtokens = []
        for idx, subtoks in enumerate(subtokens):
            for subtok in subtoks:
                indexed_subtokens.append((idx, subtok))

        return indexed_subtokens


    def unimorph_feature_parser(self, line: List[str], i: int) -> Dict[str, str]:
        if line[i] == "_":
            return {}

        return parse_unimorph_features(line[i].split(";"))


    def merge_attributes(self, tokens: List[str], value_to_attr_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Returns a dictionary containing Unimorph attributes, and the values taken on after the merge.
        """
        # First, build a list that naively merges everything
        merged_attributes: Dict[str, List[str]] = {}
        for t in tokens:
            for attr, val in t["um_feats"].items():
                if attr not in merged_attributes:
                    merged_attributes[attr] = []

                merged_attributes[attr].append(val)

        # Second, remove attributes with multiple values (even if they are the same)
        final_attributes: Dict[str, str] = {}
        for attr, vals in merged_attributes.items():
            if len(vals) == 1:
                final_attributes[attr] = vals[0]

        return final_attributes
    