from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from datasets import Dataset
from flask import Flask, request, abort
from pathlib import Path
import torch
import random
import json

# load models
device = 'cuda'
model_path = 'facebook/dpr-ctx_encoder-single-nq-base'
ctx_encoder = DPRContextEncoder.from_pretrained(model_path)
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_path)
q_encoder = DPRQuestionEncoder.from_pretrained(model_path)
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_path)
ctx_encoder.to(device)
q_encoder.to(device)

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

# precompute book embeddings
data_path = Path('data')
book_path = data_path / 'books'
dataset = load_dataset(book_path)
ds_with_embeddings = dataset.map(preprocess)
ds_with_embeddings.add_faiss_index(column='embeddings')

# load titles
title_path = data_path / 'title.json'
with open(title_path) as jsonfile:
    titles = json.load(jsonfile)

# load questions
question_path = data_path / 'questions.txt'
with open(question_path) as f:
    questions = [line.strip('\n') for line in f]

app = Flask(__name__)

@app.route("/question")
def get_question():
    return {
        'question': random.choice(questions)
    }

@app.route("/search")
def search():
    query = request.args.get('q')
    if query is None or query == '':
        abort(400, 'query is not specified')

    query = query.replace('+', ' ')
    
    query_input_ids = q_tokenizer(query, truncation=True, return_tensors='pt').input_ids
    with torch.no_grad():
        query_embedding = q_encoder(query_input_ids.to(device))[0][0].cpu().numpy()

    scores, examples = ds_with_embeddings.get_nearest_examples('embeddings', query_embedding, k=1)
    _id = examples['id'][0]
    return {
        'id': _id,
        'title': titles[_id],
        'epub_link': f'https://archive.org/download/{_id}/{_id}.pub' 
    }