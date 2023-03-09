import io
import random

import torch
import torchtext
import spacy
import collections

tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
file_path = 'human_chat.txt'
max_sentence_length = 50

def build_vocab() -> (torchtext.vocab.vocab, int):
    counter = collections.Counter()
    num_sentences = 0
    with io.open(file_path, encoding='utf8') as f:
        for idx, line in enumerate(f):
            counter.update(tokenizer(line.rstrip('\n')))
            num_sentences += 1
    return torchtext.vocab.vocab(counter, specials=['[CLS]', '[SEP]', '[PAD]', '[MASK]']), num_sentences


corpus_vocab, corpus_num_sentences = build_vocab()
corpus_itos = corpus_vocab.get_itos()
corpus_stoi = corpus_vocab.get_stoi()


def random_choose_and_pad(is_next=False) -> [int]:
    pos1 = random.randint(0, corpus_num_sentences - 1)
    pos2 = pos1 + 1 if is_next else random.randint(pos1 + 2, corpus_num_sentences)
    tokens1 = []
    tokens2 = []
    with io.open(file_path, encoding='utf8') as f:
        for idx, line in enumerate(f):
            if idx == pos1:
                for token in tokenizer(line.rstrip('\n')):
                    tokens1.append(corpus_stoi[token])
            elif idx == pos2:
                for token in tokenizer(line.rstrip('\n')):
                    tokens2.append(corpus_stoi[token])
                break
    while len(tokens1) < max_sentence_length:
        tokens1.append(corpus_stoi['[PAD]'])
    while len(tokens2) < max_sentence_length:
        tokens2.append(corpus_stoi['[PAD]'])
    return [corpus_stoi['[CLS]']] + tokens1 + [corpus_stoi['[SEP]']] + tokens2


# returns a list in shape
def generate_batch() -> list:
    batch = []
    for _ in range(int(batch_size / 2)):
        tokens = random_choose_and_pad()
        masked_tokens, masked_positions = mask_tokens(tokens)
    return batch


def mask_tokens(tokens: [int]) -> ([int], [int]):
    return [], []


# input batch should be in shape [batch_size, seq_length],
# mask tensor should be in shape [batch_size, seq_length, d_model]
def generate_attention_pad_mask(batch: torch.Tensor) -> torch.Tensor:
    mask = batch.data.eq(corpus_stoi['[PAD]']).unsqueeze(1)
    return mask.expand(-1, mask.shape[-1], -1)


d_model = 512
batch_size = 64
