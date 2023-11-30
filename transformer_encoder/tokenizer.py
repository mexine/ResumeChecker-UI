import collections
import numpy as np
import regex as re
import math
from nltk.stem import SnowballStemmer
import json
from numpy.typing import NDArray
from typing import List, Tuple


class Tokenizer:
    def __init__(self, max_pos: int, vocab_size: int = None, oov_token: int = 0) -> None:
        self.max_pos = max_pos
        self.vocab_size = vocab_size 
        self.oov_token = oov_token
        self.stemmer = SnowballStemmer('english')

    def fit_on_texts(self, text: List[List[str]]) -> None:
        """
        builds vocabulary according to text, limits vocabulary according to vocab_size\n
        accepts 2D array of word sequences, ex: [['word', 'word1'], ['word2', 'word3']]
        """
        # flatten list
        text = [word for sequence in text for word in sequence.split(' ')]

        # count every word occurence
        word_count = collections.Counter(text)

        # sort by occurence
        word_index = ['[mask]', '[pad]'] + sorted(word_count, reverse=True)

        # splice according to vocabulary size
        if (self.vocab_size):
            word_index = word_index[0:self.vocab_size]

        # convert to dict type
        self.word_index = {value: index + 1 for index, value in enumerate(word_index)}

    def clean(self, resume: str) -> str:
        resume = resume.lower()
        
        # remove special characters (e.g., punctuations)
        # special_characters = r"[\[\]!\"#$%&()*+,/:;<=>?@^_{|}~•·◦]"
        not_allowed_characters = r"[^a-z0-9\-\.']"
        resume = re.sub(not_allowed_characters, ' ', resume)

        # remove year tokens
        resume = re.sub(r"(20\d\d|19\d\d)", ' ', resume)

        # substitute numbers to [NUMBER] token
        resume = re.sub(r"\d+\.?\d*\+?|\d+th|\d+rd|\d+k", '[NUMBER]', resume)

        # remove white spaces
        resume = str.strip(re.sub(r"\s+", ' ', resume))

        new_resume = ''
        for word in resume.split(' '):
            if (re.search(r"[\.\-]", word)):
                # handler of abbreviations that use '.' (e.g., B.S.)
                if (not re.match(r"([a-z]{1,2}\.)+", word)):
                    word = re.sub(r"\.", ' ', word)
                # handler of misplaced '-' and avoid removing '-' for hypenated words (e.g., Mother-in-law)
                if (not re.match(r"[\w]+(-[\w]+)+", word)):
                    word = re.sub(r"-", ' ', word)

            # stemming
            word = self.stemmer.stem(word.strip())

            # explicit removal of observed noise words
            noise_words = ['n a', 'company name']
            if (word not in noise_words):
                new_resume += word + ' '

        return new_resume.strip()

    def truncate(self, resume: str):
        splitted_resume = resume.split(' ')

        # truncate sentence that are longer than max_pos
        if (len(splitted_resume) > self.max_pos):
            num_split = math.ceil(len(splitted_resume) / self.max_pos)
            # Split the list into n parts
            return [splitted_resume[i * self.max_pos: (i + 1) * self.max_pos] for i in range(num_split)]
        else:
            return [splitted_resume]
    
    def tokenize(self, text: List[List[str]]) -> List[List[int]]:
        """
        accepts 2D array of words, ex: [['word', 'word1'], ['word2', 'word3']]\n
        returns 2D numpy array of token ids, 0 is assigned for out-of-vocabulary words
        """
        tokenized_text = list(text)
        for index, sequence in enumerate(tokenized_text):
            tokenized_text[index] = (np.vectorize(lambda word: self.word_index.get(word, self.oov_token))(sequence).tolist())

        return tokenized_text
    
    def pad_sequences(self, text: List[List[int]], max_len: int, pad_token: int = 0) -> List[List[int]]:
        for index in range(len(text)):
            text[index] += ([pad_token] * (max_len - len(text[index])))

        return text

    def generate_attention_mask(self, text: List[List[int]], mask_token_id: int) -> np.ndarray[List[int]]:
        # create the attention mask
        attention_mask = np.array(text, dtype=np.float64)
        # change mask tokens to 0
        attention_mask[attention_mask == mask_token_id] = 0
        # change non-masked tokens to 1
        attention_mask[attention_mask > 0] = 1

        return attention_mask

    def generate_mlm_mask(self, attention_mask: List[NDArray[np.float64]]):
        # create reversed attention mask
        mlm_mask = attention_mask.copy()
        mlm_mask[mlm_mask == 1] = -1
        mlm_mask[mlm_mask == 0] = 1
        mlm_mask[mlm_mask == -1] = 0
        mlm_mask = np.expand_dims(mlm_mask, axis=-1)

        return mlm_mask
    
    def tokenize_pad_atten(self, 
                      tokens: list[list[str]]) -> Tuple[List[List[int]], np.ndarray[List[int]]]:
        """
            Should only be used during training. Also provide the additional code after MLM mask generation:

                attention_mask[attention_mask == -1] = 0

                padded_tokenized_tokens = np.array(padded_tokenized_tokens)

                padded_tokenized_tokens[padded_tokenized_tokens == -1] = self.get_pad_token_id()
        """
        tokenized_tokens = self.tokenize(tokens)
        padded_tokenized_tokens = self.pad_sequences(tokenized_tokens, self.max_pos, pad_token=-1)
        attention_mask = self.generate_attention_mask(padded_tokenized_tokens, self.get_mask_token_id())

        return (padded_tokenized_tokens, attention_mask)
    
    def clean_truncate_tokenize_pad_atten(self, 
                      resume: str) -> Tuple[List[List[int]], np.ndarray[List[int]]]:
        """
            Should only be used during production.
        """
        cleaned_resume = self.clean(resume=resume)
        truncated_tokens = self.truncate(resume=cleaned_resume)
        tokenized_tokens = self.tokenize(truncated_tokens)
        padded_tokenized_tokens = self.pad_sequences(tokenized_tokens, self.max_pos, pad_token=-1)
        attention_mask = self.generate_attention_mask(padded_tokenized_tokens, self.get_mask_token_id())

        # change padding tokens to 0
        attention_mask[attention_mask == -1] = 0
        padded_tokenized_tokens = np.array(padded_tokenized_tokens)
        padded_tokenized_tokens[padded_tokenized_tokens == -1] = self.get_pad_token_id()

        return (padded_tokenized_tokens, attention_mask)
    
    def get_mask_token_id(self):
        return self.word_index['[mask]']
    
    def get_pad_token_id(self):
        return self.word_index['[pad]']

    def save_word_index(self):
        with open('tokenizer_word_index.txt', 'w') as file:
            json.dump(self.word_index, file, indent=4)

    def load_word_index(self):
        with open('transformer_encoder/tokenizer_word_index.txt', 'r') as file:
            self.word_index = json.load(file)