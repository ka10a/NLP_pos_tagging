#!/bin/bash

pip install -r requirements.txt
wget https://www.dropbox.com/s/vttjivmmxw7leea/ru_vectors_v3.bin
mv ru_vectors_v3.bin ./data/ru_vectors_v3.bin
wget https://www.dropbox.com/s/tos2sj0cifpyds2/bilstm_pos_tagger29.ckpt
mv bilstm_pos_tagger29.ckpt ./checkpoints/model_6/bilstm_pos_tagger29.ckpt