from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import Dataset
from pathlib import Path
import torch
import argparse

# load models
device = 'cuda'
model_path = 'facebook/dpr-ctx_encoder-single-nq-base'
ctx_encoder = DPRContextEncoder.from_pretrained(model_path)
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_path)
ctx_encoder.to(device)

def preprocess(example):
    input_ids = ctx_tokenizer(example["text"], truncation=True, return_tensors="pt").input_ids
    with torch.no_grad():
        pooler_output = ctx_encoder(input_ids.to(device))[0][0].cpu().numpy()
    return {'embeddings': pooler_output}

def load_dataset(data_path):
    data_dict = {'id': [], 'text': []}
    
    for book_path in data_path.glob('*.txt'): 
        with open(book_path) as f:
            data_dict['id'].append(book_path.stem)
            data_dict['text'].append(f.read())

    dataset = Dataset.from_dict(data_dict)
    return dataset

def main(args):
    # precompute book embeddings
    dataset = load_dataset(Path(args.data_path))
    ds_with_embeddings = dataset.map(preprocess)
    ds_with_embeddings.save_to_disk(args.dataset_save_path)
    ds_with_embeddings.add_faiss_index(column='embeddings')
    ds_with_embeddings.save_faiss_index('embeddings', args.index_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/books')
    parser.add_argument('--dataset_save_path', default='./book_dataset')
    parser.add_argument('--index_save_path', default='./book_index.faiss')
    args = parser.parse_args()
    main(args)