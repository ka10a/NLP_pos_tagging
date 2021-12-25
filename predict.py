import torch
import argparse
import nltk
from nltk.tokenize import word_tokenize


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

def predict(args):
    sentence = args['sent']
    model.load_state_dict(torch.load(args['model_chpt_path']))
    model.eval()
    ft = fasttext.load_model('./data/ru_vectors_v3.bin')
    
    nltk.download('punkt')
    sent_words = word_tokenize('sentence')
    embs = [ft.get_word_vector(w) for w in sent_words]
    
    X_batch = torch.FloatTensor(X_batch)
    logits = model(X_batch.cpu())
    pred = torch.argmax(logits, dim=-1)
    
    f_tags = open('./data/tags.txt', 'r')
    id2tag = dict()
    for line in f_tags.readlines():
        i, tag = line.strip().split()
        id2tag[i] = tag
    ans = []
    for i, tag_id in enumerate(pred):
        ans.append((sent_words[i], id2tag[tag_id]))
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
    for elem in ans:
        print(elem[0], elem[1])