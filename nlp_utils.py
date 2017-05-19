# coding: utf-8

from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize

def create_word_dict(text,
                     tokenizer=word_tokenize,
                     idx_start_from=0,
                     max_vocab_size=30000,
                     at_least=5,
                     mandatory=None,
                     signals=None):
    """
    Args:
        text: input str
        tokenizer: tokenizer applied to the text
        idx_start_from: minimum index of the word dictionary
        max_vocab_size: maximun number of vacaburary
        at_least: minimum frequency of the words
        mandatory: mandatory words
    """
    if signals is None:
        signals = ["<OOV>", "<BOS>", "<EOS>"]
    if mandatory is None:
        mandatory = []
    token_count = Counter(tokenizer(text))
    word_dict = defaultdict(lambda: len(word_dict) + idx_start_from)
    most_common_token = signals + mandatory
    most_common_token.extend([x[0] for x in token_count.most_common(max_vocab_size) if x[1] >= at_least])
    vocab_size = max([word_dict[token] for token in most_common_token]) - idx_start_from + 1
    word_dict.default_factory = lambda: word_dict[signals[0]]

    return word_dict, vocab_size
