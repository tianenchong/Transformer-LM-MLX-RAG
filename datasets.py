import argparse
import io
import itertools
import math
import os
import tarfile
from typing import Dict, List
import zipfile
from urllib import request
import requests
import sentencepiece as spm
import numpy as np
import json
from mlx_lm import load, generate
import unicodedata

PAD = 0
BOS = 1
EOS = 2
UNK = 3

def qa_json_to_txt(to_save_dir):
    sub_dir = '/qa'
    to_save_dir += sub_dir
    os.makedirs(to_save_dir, exist_ok=True)

    # Loop through all files in the input folder
    file_identifier = "wikipedia-train-simplified"

    input_path = os.path.join(to_save_dir, file_identifier+".json")
    output_path = os.path.join(to_save_dir, file_identifier+".txt")
    if os.path.exists(input_path):
        # Read, lowercase, and write
        with open(input_path, "r", encoding="utf-8") as input_file:
            with open(output_path, "w", encoding="utf-8") as output_file:
                qa = json.load(input_file)
                for entry in qa['Data']:
                    output_file.write(entry['Question'] + '\n')
                    output_file.write(entry['Answer'] + '\n')

        print(f"Processed {file_identifier+".json"}")

def to_lowercase_files(from_dir, to_save_dir):
    sub_dir = '/evidence/wikipedia'
    to_save_dir += sub_dir
    from_dir += sub_dir
    # Create output folder if it doesn't exist
    os.makedirs(to_save_dir, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(from_dir):
        if filename.endswith(".txt"):  # only process .txt files
            input_path = os.path.join(from_dir, filename)
            output_path = os.path.join(to_save_dir, filename)

            # Read, lowercase, and write
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read().lower()  # lowercase

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Processed {filename}")

def train_tokenizer(to_save_dir):
    root_dir = to_save_dir
    sub_dir = '/evidence/wikipedia'
    to_save_dir += sub_dir
    qa_dir = root_dir+"/qa"
    txt_files = [os.path.join(to_save_dir, f) for f in os.listdir(to_save_dir) if f.endswith(".txt")] + [os.path.join(qa_dir, f) for f in os.listdir(qa_dir) if f.endswith(".txt")]
    target_dir = root_dir+"/tokenizer"
    os.makedirs(target_dir, exist_ok=True)
    agg_file_path = target_dir+"/files.txt"
    with open(agg_file_path, "w", encoding="utf-8") as output_file:
        for path in txt_files:
            with open(path, "r", encoding="utf-8") as input_file:
                text = input_file.read()
                output_file.write(text + "\n")  # one path per line
    spm.SentencePieceTrainer.Train(
    f"--input={agg_file_path} "
    f"--model_prefix={target_dir+"/wiki"} "
    f"--vocab_size=10000 "
    f"--max_sentence_length=10000 "
    f"--character_coverage=1.0 "
    f"--model_type=bpe "
    f"--pad_id=0 "
    f"--bos_id=1 "
    f"--eos_id=2 "
    f"--unk_id=3 "
    )


def prepare_generator_qa_dataset(from_dir="/tmp", to_save_dir="/tmp", subset = False):
    from_qa_dir = from_dir+"/qa"
    from_qa_filename = 'wikipedia-train.json'
    to_qa_dir = to_save_dir+"/qa"
    to_qa_filename = 'wikipedia-train-simplified.json'

    def download_dataset(from_dir="/tmp"):
        # URL of the file
        url = "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"
        os.makedirs(from_dir, exist_ok=True)

        # Local filename
        filename = os.path.join(from_dir, "triviaqa-rc.tar.gz")

        if not os.path.isfile(filename):
            # Download the file
            print(f"Downloading {url} ...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded to {filename}")

            # Extract the tar.gz file
            print("Extracting...")
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=from_dir)
            print(f"Extraction completed. Files are in {from_dir}")

    if not subset:
        download_dataset(from_dir)
    model, tokenizer = load("lmstudio-community/LFM2.5-1.2B-Instruct-MLX-8bit")

    with open(os.path.join(from_qa_dir, from_qa_filename), "r", encoding="utf-8") as input_file:
        qa = json.load(input_file)
        output = {"Data":[]}
        for entry in qa["Data"]: # question might need recurring embedding
            new_entry = {"EntityPages":[]}

            new_entry["EntityPages"] = [file["Filename"] for file in entry["EntityPages"]]
            new_entry["Question"] = entry['Question'].lower()

            answer = generate(model, tokenizer, prompt=
                f"Expand the following answer into a complete, clear sentence, keeping it short, with no additional information.\n\n"
                f"Question: {entry['Question']}\n"
                f"Answer: {entry['Answer']['Value']}\n\n"
                f"Expanded answer:", verbose=True)

            new_entry["Answer"] = entry['Answer']['Value'].lower()
            new_entry["FullAnswer"] = answer.strip().lower()
            output["Data"].append(new_entry)
        
        os.makedirs(to_qa_dir, exist_ok=True)
        with open(os.path.join(to_qa_dir, to_qa_filename), "w", encoding="utf-8") as output_file:
            output_file.write(json.dumps(output))

def load_retriever_dataset(window_size, to_save_dir="/tmp", generator_inferencing = False):
    sub_dir = '/evidence/wikipedia'
    doc_dir = to_save_dir + sub_dir
    qa_dir = to_save_dir+"/qa"

    def _load(qa_filename, filenames):
        spm_model = spm.SentencePieceProcessor(model_file=to_save_dir+'/tokenizer/wiki.model')
        # chunk_id -> file_id -> question_id (make sure they match up with current question for positive match, else negative match)
        # question_id -> file_ids -> chunk_ids
        chunks = [] # chunk -> file_id
        files = {} # each key is file_id -> question and chunks
        questions = [] # each question row corresponds to each answer row
        answers = [] # each answer row corresponds to each question row
        def doc_to_array(file_name, file_id, chunks):
            with open(os.path.join(doc_dir, file_name), "r") as fid:
                lines = (l.strip() for l in fid.readlines())
            tokenized_file = np.array(
                [w for line in lines for w in [BOS]+spm_model.encode(line)+[EOS]],
                dtype=np.uint32,
            )
            T = tokenized_file.size
            N = math.ceil(T/window_size)
            pad_len = N * window_size - T
            padded_tokenized_file = np.pad(tokenized_file, (0, pad_len), 'constant', constant_values=(0, 0))
            tokenized_file_chunks = padded_tokenized_file.reshape((N, window_size))
            prior_chunks_len = len(chunks)
            chunks += [{"file_id":file_id, "tokenized_chunk":chunk} for chunk in tokenized_file_chunks]
            if not generator_inferencing:
                files[file_id] = {"file_name": file_name, "chunk_ids":np.arange(prior_chunks_len, prior_chunks_len+tokenized_file_chunks.shape[0]), "question_ids":[]}
        
        def qa_to_array(qa_filename, filenames):
            with open(os.path.join(qa_dir, qa_filename), "r", encoding="utf-8") as input_file:
                qa = json.load(input_file)
                for id, entry in enumerate(qa['Data']): # question might need recurring embedding
                    _file_ids = [filenames.index(file.lower()) for file in entry["EntityPages"]]
                    questions.append({"tokenized": spm_model.encode(entry['Question']), "file_ids":_file_ids})
                    for _file_id in _file_ids:
                        files[_file_id]["question_ids"].append(id)
                    answers.append(spm_model.encode(entry['Answer']))

        total_filenames = len(filenames)
        for file_id, file_name in enumerate(filenames):
            print('processing files '+str(file_id)+' / '+str(total_filenames), end='\r')
            doc_to_array(file_name, file_id, chunks)

        if not generator_inferencing:
            qa_to_array(qa_filename, filenames)

        return chunks, files, questions, answers
    filenames = [unicodedata.normalize("NFC",f).lower() for f in os.listdir(doc_dir) if f.endswith(".txt")]
    qa_filename = qa_dir + '/wikipedia-train-simplified.json'
    return _load(qa_filename, filenames)

def load_generator_dataset(to_save_dir="/tmp"):
    sub_dir = '/evidence/wikipedia'
    doc_dir = to_save_dir + sub_dir
    qa_dir = to_save_dir+"/qa"
    
    def _load(qa_filename):
        spm_model = spm.SentencePieceProcessor(model_file=to_save_dir+'/tokenizer/wiki.model')
        # chunk_id -> file_id -> question_id (make sure they match up with current question for positive match, else negative match)
        # question_id -> file_ids -> chunk_ids
        qas = []
        qas_pad_len = None
        max_len = 0
        with open(os.path.join(qa_dir, qa_filename), "r", encoding="utf-8") as input_file:
            qa = json.load(input_file)
            qas_pad_len = np.zeros((len(qa['Data']), 2), dtype=np.int64)
            for i, entry in enumerate(qa['Data']): # question might need recurring embedding
                tokenized_answer = [BOS]+spm_model.encode(entry['FullAnswer'])+[EOS]
                tokenized_question = spm_model.encode(entry['Question'])
                tokenized_question_len = len(tokenized_question)
                tokenized_answer_len = len(tokenized_answer)
                qas_pad_len[i] = [tokenized_question_len, tokenized_answer_len]
                current_max_len = max(tokenized_question_len, tokenized_answer_len)
                max_len = max(max_len, current_max_len)
                qas.append({"q": tokenized_question, "a":tokenized_answer})
        next_highest_len = 2**math.ceil(math.log2(max_len))
        qas_pad_len = next_highest_len - qas_pad_len
        qas_out = np.array([ [qa["q"]+[0]*qas_pad_len[i][0], qa["a"]+[0]*qas_pad_len[i][1]] for i, qa in enumerate(qas)], dtype=np.int32)
        return qas_out
    qa_filename = qa_dir + '/wikipedia-train-simplified.json'
    return _load(qa_filename)

if __name__ == "__main__":
    pass