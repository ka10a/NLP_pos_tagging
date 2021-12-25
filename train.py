import fasttext
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import argparse
import nltk
from nltk.tokenize import word_tokenize


class Trainer:
    def __init__(self, model, 
                 train_X, train_y, val_X, val_y,
                 batch_size=64, lr=1e-4,
                 project="bilstm_pos_tagger", save_path='./'):
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.model = model
        self.save_path = save_path
        self.name = project
        self.batch_size = batch_size

    def make_epoch(self, X, y, train=True):
        total_loss = 0
        correct = 0
        total = 0

        if train:
            self.model.train(True)
        else:
            self.model.train(False)
        iters = np.ceil(len(X)/self.batch_size) 
        for X_batch, y_batch in tqdm(
            batch_generator(X, y, batch_size, shuffle=True), 
            total=iters):
            
            X_batch, y_batch = collate_fn_lm(X_batch, y_batch)
            y_batch = torch.LongTensor(y_batch)
            X_batch = torch.FloatTensor(X_batch)
            
            logits = self.model(X_batch.cpu())
            loss = self.loss_func(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            mask = (y_batch != 0).to(torch.long)
            pred = torch.argmax(logits, dim=-1)
            correct += ((pred == y_batch)*mask).sum().item()
            total += mask.sum().item()
            total_loss += loss.item()
        return total_loss, correct / total
    
    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.name + str(epoch) + '.ckpt'))
    
    def fit(self, max_epochs=20):
        for epoch in range(max_epochs):
            self.checkpoint(epoch)
            print('\rEpoch: %d' % epoch)
            train_loss, train_acc = self.make_epoch(self.train_X, self.train_y)
            test_loss, test_acc = self.make_epoch(self.val_X, self.val_y, train=False)
            print('Train loss: {}'.format(train_loss))
            print('Val loss: {}'.format(test_loss))
            print('Train accuracy: {}'.format(train_acc))
            print('Val accuracy: {}'.format(test_acc))

class BiLSTM_pos_tagger_v2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, target_size):
        super(BiLSTM_pos_tagger_v2, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pca = nn.Linear(embedding_dim, embedding_dim // 2)
        self.lstm = nn.LSTM(embedding_dim // 2, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, target_size)
    
    def forward(self, word_embeddings):
        pca_emb = self.pca(word_embeddings)
        lstm_out, _ = self.lstm(pca_emb)
        out = self.fc(lstm_out)
        out = self.dropout(out)
        tag_space = self.fc2(out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def batch_generator(X, y, batchsize, shuffle=True):
    '''
        Генерирует tuple из батча объектов и их меток
    '''
    indices = np.arange(len(X))
    if shuffle:
        indices = np.random.permutation(indices)
    # идем по всем данным с шагом batchsize
    # возвращаем start: start + batchsize объектов на каждой итерации
    for start in range(0, len(indices), batchsize):
        idxs = indices[start:(start + batchsize)]
        X_tmp = [X[i] for i in idxs]
        y_tmp = [y[i] for i in idxs]  
        yield X_tmp, y_tmp

def collate_fn_lm(samples_X, samples_y):
    batch_size = len(samples_X)
    max_len = max(len(sample) for sample in samples_X)
    emb_len = len(samples_X[0][0])

    src_tensor_X = torch.zeros((batch_size, max_len, emb_len), dtype=torch.float)
    src_tensor_y = torch.zeros((batch_size, max_len), dtype=torch.long)

    for batch_id, s in enumerate(samples_y):
        for i, elem in enumerate(s):
            src_tensor_X[batch_id][i][:] = torch.tensor(elem)
    for (batch_id, s) in enumerate(samples_y):
        length = len(s)
        src_tensor_y[batch_id][:length] = torch.tensor(s)

    return src_tensor_X, src_tensor_y

def train(args):
    ft = fasttext.load_model('./data/ru_vectors_v3.bin')

    f = open('./data/ru_pud-ud-test.conllu', 'r')
    sent_info = []
    sentences, postags, embeddings = [], [], []
    for line in f.readlines():
        if len(line) < 2:
            i = 0
            while '# text = ' not in sent_info[i]:
                i += 1
            sent = sent_info[i][9:]
            pos = []
            embs = []
            for j in range(i + 2, len(sent_info)):
                tmp = sent_info[j].split('\t')
                pos.append(tmp[3])
                embs.append(ft.get_word_vector(tmp[1]))
            sentences.append(sent)
            postags.append(pos)
            embeddings.append(embs)
            sent_info = []
        else:
            sent_info.append(line)
    f.close()

    tags = set()
    for elem in postags:
        tags.update(elem)
    tag2idx = dict()
    idx2tag = ['-'] + list(tags)
    for i, t in enumerate(idx2tag):
        tag2idx[t] = i

    postags_idx = [[tag2idx[tag] for tag in sent] for sent in postags]
    train_X, test_X, train_y, test_y = train_test_split(embeddings, postags_idx, 
                                                        shuffle=True,
                                                        random_state=42,
                                                        test_size=args['val_size'])

    batch_size = args['batch_size']
    num_layers = args['num_layers']
    hidden_size = args['hidden_size']
    embedding_length = ft.get_dimension()
    target_size = len(idx2tag)
    save_path = args['checkpoint_path']

    my_model = BiLSTM_pos_tagger_v2(embedding_length, hidden_size, num_layers, target_size)
    trainer1 = Trainer(my_model, train_X, train_y, test_X, test_y, batch_size=batch_size, save_path=save_path, lr=1e-4)
    trainer1.fit(max_epochs=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the POS tagging model')
    parser.add_argument('--checkpoint_path', action='store',
                        default='./checkpoints/',
                        help='Checkpoint path',
                        help='Train set size')
    parser.add_argument('--val_size', action='store',
                        default=0.3, type=float,
                        help='Val set size')
    parser.add_argument('--batch_size', action='store',
                        default=32, type=int,
                        help='Batch size')
    parser.add_argument('--num_layers', action='store',
                        default=4, type=int,
                        help='Num layers')
    parser.add_argument('--hidden_size', action='store',
                        default=300, type=int,
                        help='Hidden dim')
    parser.add_argument('--epochs', action='store',
                        default=30, type=int,
                        help='Epochs')
    parser.add_argument('--lr', action='store',
                        default=1e-4, type=float,
                        help='Learning rate')
    args = vars(parser.parse_args())
    train(args)