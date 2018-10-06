import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import deque


class CorpusList():
    def __init__(self, wordlist):
        self.word_pair_catch = deque()


class SkipGramModel(nn.Module):
    def __init__(self, embed_size, embed_dims, window=5, batch_size=86, lr=0.025):
        super(SkipGramModel, self).__init__()
        self.embed_size = embed_size
        self.embed_dims = embed_dims
        self.u_embeds = nn.Embedding(2 * embed_size - 1, embed_dims, sparse=True)
        self.v_embeds = nn.Embedding(2 * embed_size - 1, embed_dims, sparse=True)
        self.window = window
        self.batch_size = batch_size
        self.start_lr = lr
        self.lr = lr
        tree = HuffmanTree(self.word_frequency)
        self.optimizer = optim.SGD(self.parameters(), lr=0.025)
        self.init_emb()

    def init_samples(self):
        self.table = []
        table_size = 1e6
        word_freq = np.array(list(self.word_freq.values()))**0.75
        words = sum(word_frequency)
        ratio = word_freq / words
        count = numpy.round(ratio * table_size)
        for wordid, counter in enumerate(count):
            self.sample_table += [wordid] * int(counter)
        self.table = np.array(self.table)

    def init_emb(self):
        initrange = 0.5 / self.embed_dims
        self.u_embeds.weight.data.uniform_(-initrange, initrange)
        self.v_embeds.weight.data.uniform_(-0, 0)

    def train_model(self):
        pair_count = self.data.evaluate_pair_count(self.window)
        batch_count = self.iteration * pair_count / self.batch_size
        iterations = tqdm(range(int(batch_count)))
        self.save_embedding(self.data.id2word, self.window)
        pairs = self.data.get_batch_pairs(self.batch_size, self.window)
        pairs, neg_pairs = self.data.get_batch_pairs(pairs, 5)
        for x in iterations:
            pairs = self.data.get_batch_pairs(self.batch_size, self.window)
            u = [int(pair[0]) for pair in pairs]
            v = [int(pair[1]) for pair in pairs]
            neg_u = [int(pair[0]) for pair in neg_pairs]
            neg_v = [int(pair[1]) for pair in neg_pairs]

            self.optimizer.zero_grad()
            loss = self. forward(u, v, neg_u, neg_v)
            loss.backward()
            self.optimizer.step()
            iterations.set_description("loss = %0.4f, lr = %0.8f" % (loss.data[0], self.optimizer.param_groups[0]['lr']))
            if x * self.batch_size % 100000 == 0:
                self.lr = self.start_lr
                for group in self.optimizer.param_groups:
                    group['lr'] = self.lr
        self.save_embedding(sellf.data.id2word, self.outfilename)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        embed_u = self.u_embeddings(Variable(torch.LongTensor(pos_u)))
        embed_v = self.v_embeddings(Variable(torch.LongTensor(pos_v)))
        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))
        neg_embed_u = self.u_embeddings(Variable(torch.LongTensor(neg_u)))
        neg_embed_v = self.v_embeddings(Variable(torch.LongTensor(neg_v)))
        neg_score = torch.mul(neg_embed_u, neg_embed_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)

    def save_embedding(self, id2word, file_name):
        embeds = self.u_embeds.weight.data.numpy()
        fout = open(file_name, 'w', encoding="UTF-8")
        fout.write('%d %d\n' % (len(id2word), self.embed_dims))
        for wordid, w in id2word.items():
            e = embeds[wordid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

if __name__ == '__main__':
    inputfilename = "./data/corpus.txt"
    outputfilename = "embeds.txt"
    model = SkipGramModel(128, 100)
    model.train_model()