import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import fasttext


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

def collate_fn_lm(samples_X):
    batch_size = len(samples_X)
    max_len = max(len(sample) for sample in samples_X)
    emb_len = len(samples_X[0][0])

    src_tensor_X = torch.zeros((batch_size, max_len, emb_len), dtype=torch.float)

    for batch_id, s in enumerate(samples_X):
        for i, elem in enumerate(s):
            src_tensor_X[batch_id][i][:] = torch.tensor(elem)

    return src_tensor_X
    
def predict(args):
    f_tags = open('./data/tags.txt', 'r')
    id2tag = dict()
    for line in f_tags.readlines():
        i, tag = line.strip().split()
        id2tag[int(i)] = tag
    
    ft = fasttext.load_model('./data/ru_vectors_v3.bin')
    model = BiLSTM_pos_tagger_v2(num_layers=4, hidden_dim=300, embedding_dim=ft.get_dimension(), target_size=len(id2tag.keys()))
    model.load_state_dict(torch.load(args['model_chpt_path']))
    model.eval()
    
    
    sentence = args['sent']
    nltk.download('punkt')
    sent_words = [word_tokenize(t) for t in sent_tokenize(sentence)]
    embs = [[ft.get_word_vector(w) for w in s] for s in sent_words]
    
    X_batch = torch.FloatTensor(collate_fn_lm(embs))
    logits = model(X_batch.cpu())
    pred = torch.argmax(logits, dim=-1).numpy()
    
    ans = []
    for j, s in enumerate(pred):
        tmp = []
        for i, tag_id in enumerate(s):
            if i < len(sent_words[j]):
                tmp.append((sent_words[j][i], id2tag[tag_id]))
        ans.append(tmp)
    return ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict POS tags')
    parser.add_argument('--model_chpt_path', action='store',
                        default='./checkpoints/model_6/bilstm_pos_tagger29.ckpt',
                        help='POS tagger checkpoint')
    parser.add_argument('--sent', action='store',
                        default='Мама мыла раму',
                        help='Sentence')

    args = vars(parser.parse_args())
    ans = predict(args)
    for s in ans:
        for elem in s:
            print(elem[0], elem[1])