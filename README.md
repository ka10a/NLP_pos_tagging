# NLP_pos_tagging
Homework for NLP course (Autumn 2021)

To start working with models do
> chmod 777 init.sh

> ./init.sh

How to use:
> train.py --checkpoint_path <where save checkpoints> --val_size <size of validation, float> --batch_size <batch size> --num_layers <num_layers in bilstm> --hidden_size <hidden_dim in bilstm>  --epochs <number of epochs> --lr <lr for optimizer>

> predict.py --model_chpt_path <path to checkpoint> --sent <your sentence>

Examples:
> python train.py --checkpoint_path ./checkpoints/test/ --epochs 3

> python predict.py --sent "Я люблю мороженое. А Петя любит борщ."
    
To achive good accuracy train models on 25-35 epochs and use this hyperparameters: --num_layers 4 --hidden_size 500 --lr 0.0001

Models are tuned BiLSTM on fasttext embeddings.
    
To see how models learn visit https://wandb.ai/imroggen/bilstm_pos_tagger
