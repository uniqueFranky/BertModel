import io
import math
import random

import torch
import torchtext
import spacy
import collections

tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
file_path = 'human_chat.txt'
max_sentence_length = 200


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
    num_mask = min(int(0.15 * len(tokens)), 15)
    masked_positions = [i for i in range(0, len(tokens)) if tokens[i] > 3]
    random.shuffle(masked_positions)
    masked_positions = masked_positions[: num_mask]
    masked_tokens = [tokens[pos] for pos in masked_positions]
    for pos in masked_positions:
        if random.random() < 0.8:
            tokens[pos] = corpus_stoi['[MASK]']
        elif random.random() < 0.5:
            tokens[pos] = random.randint(corpus_stoi['[MASK]'] + 1, corpus_vocab_size)
    while len(masked_positions) < 15:
        masked_positions.append(0)
        masked_tokens.append(0)
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
        batch.append((tokens, masked_positions, masked_tokens, 0))

        tokens, masked_positions, masked_tokens = random_choose_and_pad_and_mask(is_next=True)
        batch.append((tokens, masked_positions, masked_tokens, 1))
    return batch


# input batch should be in shape [batch_size, seq_length],
# mask tensor should be in shape [batch_size, seq_length, seq_length]
def generate_attention_pad_mask(batch: torch.Tensor) -> torch.Tensor:
    mask = batch.data.eq(corpus_stoi['[PAD]']).unsqueeze(1)
    return mask.expand(-1, mask.shape[-1], -1)


# generated tokens_tensor is in shape [batch_size, seq_length]
# masked_positions_tensor is in shape [batch_size, num_pad]
# masked_tokens_tensor is in shape [batch_size, num_pad]
# cls_tensor is in shape [batch_size]

def generate_tensors(batch: list) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    token_batch = [item[0] for item in batch]
    tokens_tensor = torch.tensor(token_batch, dtype=torch.long)
    position_batch = []
    for item in batch:
        position_batch.append(item[1])
    masked_positions_tensor = torch.tensor(position_batch, dtype=torch.long)
    masked_token_batch = [item[2] for item in batch]
    masked_tokens_tensor = torch.tensor(masked_token_batch, dtype=torch.long)
    cls_batch = [item[3] for item in batch]
    cls_tensor = torch.tensor(cls_batch, dtype=torch.long)
    return tokens_tensor, masked_positions_tensor, masked_tokens_tensor, cls_tensor


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
        return context


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
        positions = positions.expand(x.shape[0], -1)

        segments = torch.cat([torch.zeros(x.shape[0], max_sentence_length + 1, dtype=torch.long),
                              torch.ones(x.shape[0], max_sentence_length + 1, dtype=torch.long)], dim=-1)
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


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Bert(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads):
        super(Bert, self).__init__()

        self.embedding = BertEmbedding(vocab_size, embedding_size)
        self.encoder_stack = [EncoderLayer(embedding_size, num_heads) for _ in range(6)]
        self.classifier = torch.nn.Linear(embedding_size, 2)
        self.tanh = torch.nn.Tanh()
        self.predictor = torch.nn.Linear(embedding_size, embedding_size)
        self.gelu = gelu
        self.decoder = torch.nn.Linear(embedding_size, vocab_size, bias=False)
        self.decoder.weight = self.embedding.token_embedding.weight
        self.decoder.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        attention_mask = generate_attention_pad_mask(x)
        x = self.embedding(x)
        for encoder in self.encoder_stack:
            x = encoder(x, attention_mask)

        cls = x[:, 0, :]
        cls.squeeze(1)
        # now cls is in shape [batch_size, embedding_size]

        cls = self.tanh(self.classifier(cls))
        # now cls is in shape [batch_size, 2], ranging from -1 to +1

        # pred is in shape [batch_size, seq_len, embedding_size]
        pred = x
        pred = self.decoder(self.gelu(self.predictor(pred)))
        return cls, pred


d_model = 512
batch_size = 64
model = Bert(corpus_vocab_size, d_model, 8)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch = generate_batch()


for epoch in range(100):
    print(f'epoch {epoch+1} / 100')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, 'model.pth')
    print('checkpoint saved')

    batch = generate_batch()
    token_tensor, masked_position_tensor, masked_token_tensor, cls_tensor = generate_tensors(batch)
    cls_pred, masked_pred = model(token_tensor)
    loss1 = criterion(cls_pred, cls_tensor)

    # masked_pred is in shape [batch_size, seq_len, vocab_size]
    # masked_position_tensor is in shape [batch_size, num_pad]
    #   and needs to be converted into [batch_size, num_pad, vocab_size]
    masked_position_tensor = masked_position_tensor[:, :, None].expand(-1, -1, corpus_vocab_size)
    masked_pred = torch.gather(masked_pred, 1, masked_position_tensor)

    loss2 = criterion(masked_pred.transpose(1, 2), masked_token_tensor)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
