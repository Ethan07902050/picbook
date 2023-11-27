#!/bin/bash

rm -f model_store/dpr.mar

# List of strings
extra_files=( 
"./dpr_model/book_dataset.zip"
"./dpr_model/book_index.faiss" 
"./dpr_model/config.json"
"./dpr_model/special_tokens_map.json"
"./dpr_model/tokenizer_config.json"
"./dpr_model/vocab.txt")

# Set the IFS to ','
IFS=','

# Join the list with ','
extra_files_string="${extra_files[*]}"

# Reset IFS to its default value (space, tab, and newline)
IFS=$' \t\n'

# Print the joined string
echo $extra_files_string

torch-model-archiver --model-name "dpr" \
    --version 1.0 \
    --serialized-file ./dpr_model/model.safetensors \
    --extra-files $extra_files_string \
    --handler "./transformer_handler.py" \
    --export-path model_store