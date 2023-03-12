import io
import math
import random

import torch
import torchtext
import spacy
import collections

tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
file_path = 'human_chat.txt'
max_sentence_length = 150


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
corpus_vocab_size = len(corpus_stoi)


def mask(tokens: [int]) -> ([int], [int]):
    num_mask = int(0.15 * len(tokens))
    masked_positions = [i for i in range(0, len(tokens)) if tokens[i] > 3]
    random.shuffle(masked_positions)
    masked_positions = masked_positions[: num_mask]
    masked_tokens = [tokens[pos] for pos in masked_positions]
    for pos in masked_positions:
        if random.random() < 0.8:
            tokens[pos] = corpus_stoi['[MASK]']
        elif random.random() < 0.5:
            tokens[pos] = random.randint(corpus_stoi['[MASK]'] + 1, corpus_vocab_size)
    return masked_positions, masked_tokens


def random_choose_and_pad_and_mask(is_next=False) -> ([int], [int], [int]):
    # random choose
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

    # pad
    while len(tokens1) < max_sentence_length:
        tokens1.append(corpus_stoi['[PAD]'])
    while len(tokens2) < max_sentence_length:
        tokens2.append(corpus_stoi['[PAD]'])
    tokens = [corpus_stoi['[CLS]']] + tokens1 + [corpus_stoi['[SEP]']] + tokens2
    masked_positions, masked_tokens = mask(tokens)
    return tokens, masked_positions, masked_tokens


# generated batch is [(tokens, masked_positions, masked_tokens)]
def generate_batch() -> list:
    batch = []
    for _ in range(int(batch_size / 2)):
        tokens, masked_positions, masked_tokens = random_choose_and_pad_and_mask(is_next=False)
        batch.append((tokens, masked_positions, masked_tokens, False))

        tokens, masked_positions, masked_tokens = random_choose_and_pad_and_mask(is_next=True)
        batch.append((tokens, masked_positions, masked_tokens, True))
    return batch


# input batch should be in shape [batch_size, seq_length],
# mask tensor should be in shape [batch_size, seq_length, seq_length]
def generate_attention_pad_mask(batch: torch.Tensor) -> torch.Tensor:
    mask = batch.data.eq(corpus_stoi['[PAD]']).unsqueeze(1)
    return mask.expand(-1, mask.shape[-1], -1)


# generated token_tensor is in shape [batch_size, seq_length]
def generate_tensors(batch: list) -> torch.Tensor:
    token_batch = [item[0] for item in batch]
    return torch.tensor(token_batch, dtype=torch.long)


class AttentionHead(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionHead, self).__init__()

        self.w_q = torch.nn.Linear(input_size, output_size)
        self.w_k = torch.nn.Linear(input_size, output_size)
        self.w_v = torch.nn.Linear(input_size, output_size)
        self.scale = math.sqrt(output_size)

    def forward(self, x, attention_mask):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        attention_weight = torch.matmul(q, k.transpose(-1, -2))
        attention_weight /= self.scale
        attention_weight.masked_fill(attention_mask, 1e-9)
        score = torch.nn.Softmax(dim=-1)(attention_weight)
        context = torch.matmul(score, v)
        return context, attention_weight, score


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, embedding_size, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = [AttentionHead(embedding_size, d_model) for _ in range(num_heads)]
        self.linear = torch.nn.Linear(num_heads * d_model, d_model)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, attention_mask):
        contexts = [head(x, attention_mask) for head in self.heads]
        context = contexts[0]
        for i in range(1, self.num_heads):
            context = torch.cat((context, contexts[i]), -1)
        context = self.linear(context)
        context = self.norm(context + x)
        return context


class BertEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(BertEmbedding, self).__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = torch.nn.Embedding(max_sentence_length * 2 + 2, embedding_size)
        self.segment_embedding = torch.nn.Embedding(2, embedding_size)
        self.norm = torch.nn.LayerNorm(embedding_size)

    def forward(self, x):
        positions = torch.arange(0, max_sentence_length * 2 + 2, dtype=torch.long)
        positions.expand(x.shape[0], -1)

        segments = torch.cat([torch.zeros(x.shape[0], max_sentence_length + 1),
                              torch.ones(x.shape[0], max_sentence_length + 1)], dim=-1)

        embedding = self.token_embedding(x) + self.position_embedding(positions) + self.segment_embedding(segments)
        return self.norm(embedding)


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, d_model):
        super(FeedForwardNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(d_model, d_model * 4)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(d_model * 4, d_model)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.linear2(self.relu(self.linear1(x))) + x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(num_heads, embedding_size, embedding_size)
        self.feed_forward = FeedForwardNetwork(embedding_size)

    def forward(self, x, attention_mask):
        x = self.multi_head_attention(x, attention_mask)
        x = self.feed_forward(x)
        return x




d_model = 512
batch_size = 64
batch = generate_batch()
token_tensor = generate_tensors(batch)
print(token_tensor.shape)
print(generate_attention_pad_mask(token_tensor).shape)
