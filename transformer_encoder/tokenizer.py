import collections
import numpy as np
import regex as re
import math
from nltk.stem import SnowballStemmer
import json
from numpy.typing import NDArray
from typing import List, Tuple


class Tokenizer:
    # def fit_on_texts(self, text: List[List[str]]) -> None:
    #     """
    #     builds vocabulary according to text, limits vocabulary according to vocab_size\n
    #     accepts 2D array of word sequences, ex: [['word', 'word1'], ['word2', 'word3']]
    #     """
    #     # flatten list
    #     text = [word for sequence in text for word in sequence.split(' ')]

    #     # count every word occurence
    #     word_count = collections.Counter(text)

    #     # sort by occurence
    #     word_index = ['[mask]', '[pad]'] + sorted(word_count, reverse=True)

    #     # splice according to vocabulary size
    #     if (self.vocab_size):
    #         word_index = word_index[0:self.vocab_size]

    #     # convert to dict type
    #     self.word_index = {value: index + 1 for index, value in enumerate(word_index)}

    def clean(self, text: str) -> list[str]:
        text = text.lower()
        
        # remove special characters (e.g., punctuations)
        # special_characters = r"[\[\]!\"#$%&()*+,/:;<=>?@^_{|}~•·◦]"
        not_allowed_characters = r"[^\w\d.,?!:;`\'\"()\[\]{}—–+\-*/<>=&@\\]"
        text = re.sub(not_allowed_characters, ' ', text)

        # # remove year tokens
        # resume = re.sub(r"(20\d\d|19\d\d)", ' ', resume)

        # # substitute numbers to [NUMBER] token
        # resume = re.sub(r"\d+\.?\d*\+?|\d+th|\d+rd|\d+k", '[NUMBER]', resume)

        # remove white spaces
        text = str.strip(re.sub(r"\s+", ' ', text))

        # new_resume = ''
        # for word in resume.split(' '):
            # if (re.search(r"[\.\-]", word)):
            #     # handler of abbreviations that use '.' (e.g., B.S.)
            #     if (not re.match(r"([a-z]{1,2}\.)+", word)):
            #         word = re.sub(r"\.", ' ', word)
            #     # handler of misplaced '-' and avoid removing '-' for hypenated words (e.g., Mother-in-law)
            #     if (not re.match(r"[\w]+(-[\w]+)+", word)):
            #         word = re.sub(r"-", ' ', word)

            # # stemming
            # word = self.stemmer.stem(word.strip())

            # # explicit removal of observed noise words
            # noise_words = ['n a', 'company name']
            # if (word not in noise_words):
            #     new_resume += word + ' '

        # return new_resume.strip()

        # split words and numbers from special characters
        return ['[CLS]'] + re.findall(r"\b\w+\b|\d+|[^\w\s]", text) + ['[SEP]']
        # return re.findall(r"\b\w+\b|\d+|[^\w\s]", text)

    def truncate(self, tokens: list[str], max_pos: int):
        # truncate sentence that are longer than max_pos
        if (len(tokens) > max_pos):
            num_split = math.ceil(len(tokens) / max_pos)
            # Split the list into n parts
            return [tokens[i * max_pos: (i + 1) * max_pos] for i in range(num_split)]
            # return [tokens[:max_pos]]
        else:
            return [tokens]
    
    # def tokenize(self, text: List[List[str]]) -> List[List[int]]:
    #     """
    #     accepts 2D array of words, ex: [['word', 'word1'], ['word2', 'word3']]\n
    #     returns 2D numpy array of token ids, 0 is assigned for out-of-vocabulary words
    #     """
    #     tokenized_text = list(text)
    #     for index, sequence in enumerate(tokenized_text):
    #         tokenized_text[index] = (np.vectorize(lambda word: self.word_index.get(word, self.oov_token))(sequence).tolist())

    #     return tokenized_text
    
    def pad_sequences(self, text: List[List[int]], pad_token_id: int) -> List[List[int]]:
        seq_lengths = [len(sublist) for sublist in text]

        for index in range(len(text)):
            text[index] += ([pad_token_id] * (max(seq_lengths) - len(text[index])))

        return np.array(text)
    
    # revise this
    def wordpiece_tokenize(self, tokens: list[str], oov_token: str) -> list[str]: 
        wordpiece_tokenized = []

        for token in tokens:
            last_token_not_oov = True
            i = 0

            while i < len(token):
                # Find the longest subword in the vocabulary
                for j in range(len(token), i, -1):
                    if (i > 0):
                        subword = '##' + token[i:j]
                    else:
                        subword = token[i:j]
                        
                    if subword in self.vocab:
                        wordpiece_tokenized.append(self.vocab[subword])
                        last_token_not_oov = True
                        i = j
                        break
                    elif(i == (j - 1) and last_token_not_oov):
                        wordpiece_tokenized.append(self.vocab[oov_token])
                        last_token_not_oov = False
                        i = j
                        break
                    elif(i == (j - 1)):
                        i = j
                        break

        # map tokens to token IDs
        return wordpiece_tokenized


    def generate_attention_mask(self, text: List[List[int]], mask_token_id: str, oov_token_id: str) -> np.ndarray[List[int]]:
        # create the attention mask
        attention_mask = np.array(text, dtype=np.float32)
        # change mask and oov tokens to 0
        attention_mask = np.where(np.equal(attention_mask, mask_token_id), 0, attention_mask)
        attention_mask = np.where(np.equal(attention_mask, oov_token_id), 0, attention_mask)
        # change non-masked tokens to 1
        attention_mask[attention_mask > 0] = 1

        return attention_mask

    def generate_mlm_mask(self, attention_mask: List[NDArray[np.float32]]):
        # create reversed attention mask
        mlm_mask = attention_mask.copy()
        mlm_mask[mlm_mask == 1] = -1
        mlm_mask[mlm_mask == 0] = 1
        mlm_mask[mlm_mask == -1] = 0
        mlm_mask = np.expand_dims(mlm_mask, axis=-1)

        return mlm_mask
    
    # def tokenize_pad_atten(self, 
    #                   tokens: list[list[str]]) -> Tuple[List[List[int]], np.ndarray[List[int]]]:
    #     """
    #         Should only be used during training. Also provide the additional code after MLM mask generation:

    #             attention_mask[attention_mask == -1] = 0

    #             padded_tokenized_tokens = np.array(padded_tokenized_tokens)

    #             padded_tokenized_tokens[padded_tokenized_tokens == -1] = self.get_pad_token_id()
    #     """
    #     tokenized_tokens = self.tokenize(tokens)
    #     padded_tokenized_tokens = self.pad_sequences(tokenized_tokens, self.max_pos, pad_token=-1)
    #     attention_mask = self.generate_attention_mask(padded_tokenized_tokens, self.get_mask_token_id())

    #     return (padded_tokenized_tokens, attention_mask)
    
    def __call__(self, text: str, pad_token: str, mask_token: str, oov_token: str, max_pos: int) -> Tuple[List[List[int]], np.ndarray[List[int]]]:
        """
            Should only be used during production.
        """
        cleaned_resume = self.clean(text=text)
        tokenized_tokens = self.wordpiece_tokenize(tokens=cleaned_resume, oov_token=oov_token)
        truncated_tokens = self.truncate(tokens=tokenized_tokens, max_pos=max_pos)
        padded_tokenized_tokens = self.pad_sequences(text=truncated_tokens, pad_token_id=self.vocab[pad_token])
        attention_mask = self.generate_attention_mask(text=padded_tokenized_tokens, mask_token_id=self.vocab[mask_token], oov_token_id=self.vocab[oov_token])

        # change padding tokens to 0
        # attention_mask[attention_mask == -1] = 0
        # padded_tokenized_tokens = np.array(padded_tokenized_tokens)
        # padded_tokenized_tokens[padded_tokenized_tokens == -1] = self.get_pad_token_id()

        return (padded_tokenized_tokens, attention_mask)

    # def save_vocab(self):
    #     with open('tokenizer_word_index.txt', 'w') as file:
    #         json.dump(self.word_index, file, indent=4)

    def load_vocab(self, file_name: str):
        self.vocab = np.load(file_name, allow_pickle=True).item()