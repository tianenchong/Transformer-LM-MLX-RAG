from collections import deque
from enum import Enum
import hashlib
import itertools
import json
import math
import os
import random
import signal
import sys
import threading
import time
from functools import partial
from typing import Any, Callable, Dict, Optional
import dill
import numpy as np
import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open, numpy
import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser("Train a encoder-decoder RAG Transformer LM with MLX.")
parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the RNGs.")
parser.add_argument(
    "--save_dir",
    type=str,
    default="/Volumes/RAMDisk/transformer-lm-rag",
    help="Model and data directory for saving, default to /Volumes/RAMDisk/transformer-lm-rag.",
)
parser.add_argument(
    "--from_dir",
    type=str,
    default="~/Downloads/triviaqa-rc",
    help="from data root directory",
)
parser.add_argument(
    "--to_dir",
    type=str,
    default="/Volumes/RAMDisk/transformer-lm-rag",
    help="to data root directory",
)
parser.add_argument(
    "--target",
    type=str,
    choices=["none", "dataset", "retriever", "all"],
    default="none",
    help="Select target mode: 'none' (do nothing), 'dataset' (prepare dataset), 'retriever' (train retriever only), or 'all' (full RAG pipeline).",
)
parser.add_argument(
    "--context_size",
    type=int,
    default=64,
    help="Context size in tokens of the model.",
)
parser.add_argument(
    "--vocab_size",
    type=int,
    default=10000,
    help="vocab size in terms of bpm of the model.",
)
parser.add_argument(
    "--num_blocks", type=int, default=4, help="Number of Transformer blocks."
)
parser.add_argument(
    "--dim",
    type=int,
    default=1024,
    help="Dimensionality of embeddings and hidden layers.",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=8,
    help="Number of heads used for multi-head attention",
)
parser.add_argument(
    "--checkpoint", action="store_true", help="Perform gradient checkpointing"
)
parser.add_argument("--batch_size", type=int, default=32, help="Minibatch size.")
parser.add_argument(
    "--num_iters", type=int, default=100000, help="Iterations to train for."
)
parser.add_argument(
    "--learning_rate", type=float, default=3e-4, help="AdamW learning rate."
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-5, help="Set the weight decay"
)
parser.add_argument(
    "--lr_warmup", type=int, default=200, help="LR linear warmup iterations"
)
parser.add_argument(
    "--identifier", type=str, default=None, help="generator model file identifier"
)
parser.add_argument(
    "--retriever_identifier", type=str, default=None, help="retriever model file identifier"
)
parser.add_argument(
    "--embedding_identifier", type=str, default=None, help="embedding model file identifier"
)
parser.add_argument(
    "--steps_per_report",
    type=int,
    default=10,
    help="Number of training steps between loss reporting.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=50,
    help="Top-k sampling for token output.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.5,
    help="Temperature for token output after top-k sampling.",
)
parser.add_argument(
    "--steps_per_eval",
    type=int,
    default=1000,
    help="Number of training steps between validations.",
)
parser.add_argument(
    "--new_model",
    action="store_true",
    help="Start new model from scratch",
)
parser.add_argument(
    "--inference",
    action="store_true",
    help="Interactive inference mode",
)
parser.add_argument(
    "--visualisation",
    action="store_true",
    help="visualise training progress at a fixed interval",
)
args = parser.parse_args()
if not args.gpu:
    mx.set_default_device(mx.cpu)

batch_size = args.batch_size
context_size = args.context_size
steps_per_eval = args.steps_per_eval
steps_per_report = args.steps_per_report

model_folder = args.save_dir
model_file_extension = ".safetensors"
metadata_file_extension = ".metadata"

finalise_and_exit = threading.Event()
force_exit = threading.Event()
saving_in_progress = threading.Event()
lock = threading.Lock()
ctrl_c_count = 0
ctrl_c_timer = None

tokenizer_file_path = args.save_dir+'/tokenizer/wiki.model'
if os.path.isfile(tokenizer_file_path):
    spm_model = spm.SentencePieceProcessor(model_file=tokenizer_file_path)

PAD = 0
BOS = 1
EOS = 2
UNK = 3

def finalize_interrupt():
    global ctrl_c_count
    with lock:
        print("\nCtrl+C detected: will finish current job, then save and exit.")
        ctrl_c_count = 0
        finalise_and_exit.set()


def handle_ctrl_c(signum, frame):
    global ctrl_c_count, ctrl_c_timer
    with lock:
        ctrl_c_count += 1
        if ctrl_c_count == 1:
            if ctrl_c_timer:
                ctrl_c_timer.cancel()
            ctrl_c_timer = threading.Timer(1, finalize_interrupt)
            ctrl_c_timer.start()
        elif ctrl_c_count == 2:
            print(
                "\nDouble Ctrl+C detected: force exit immediately (or after save finishes)."
            )
            if ctrl_c_timer:
                ctrl_c_timer.cancel()
            force_exit.set()
            if not saving_in_progress.is_set():
                sys.exit(1)


signal.signal(signal.SIGINT, handle_ctrl_c)

def save_invalid_question_ids(question_ids = set()):
    save_dir = args.save_dir
    to_qa_dir = save_dir+"/qa"
    with open(os.path.join(to_qa_dir, "unsupported-wikipedia-train.json"), "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps({"question_ids": list(question_ids)}))

def _mx2np(mx_tuple: tuple[str, mx.array]) -> Dict[str, np.ndarray]:
    new_dict = {}
    for k, v in mx_tuple:
        new_dict[k] = np.asarray(v)
    return new_dict

def git_hash_bytes(data_bytes):
    sha1 = hashlib.sha1(data_bytes).hexdigest()
    return sha1

def save(model, optimizer, it, type="retriever", exit_after_save=False):
    saving_in_progress.set()

    saved = False
    try:
        flattened_model = tree_flatten(model.trainable_parameters())
        model = _mx2np(flattened_model)
        model_bytes = numpy.save(model)
        model_hash = git_hash_bytes(model_bytes)
        model_type_extension = ".retriever" if type=="retriever" else ".generator"
        model_extension = ".safetensors"
        metadata_extension = ".metadata"
        new_extension = ".new"
        model_metadata_path = model_folder + "/" + model_hash + model_type_extension
        flattened_optimizer_state = tree_flatten(optimizer.state)
        optimizer_state = _mx2np(flattened_optimizer_state)
        if type=="retriever":
            training_state = {
                "optimizer": optimizer_state,
                "start": it,
            }
        else:
            training_state = {
                "optimizer": optimizer_state,
                "start": it,
            }

        while not saved:
            try:
                # Critical save section: do NOT abort mid-way
                with open(
                    model_metadata_path + model_extension + new_extension,
                    "wb",
                ) as f:
                    f.write(model_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                with open(
                    model_metadata_path + metadata_extension + new_extension,
                    "wb",
                ) as f:
                    dill.dump(training_state, f)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure all data is written to disk

                safe_open(
                    model_metadata_path + model_extension + new_extension,
                    framework="np",
                )

                with open(
                    model_metadata_path + metadata_extension + new_extension,
                    "rb",
                ) as f:
                    _ = dill.load(f)

            except Exception as e:
                print("Save error:", e)
                # Check force_exit before retrying
                if force_exit.is_set():
                    print(
                        f"\r\nForce exit detected during save error handling. Aborting save."
                    )
                    break  # Abort save immediately

                # Sleep in 1 second chunks to be able to check force_exit regularly
                sleep_seconds = 300  # 5 minutes
                for _ in range(sleep_seconds):
                    if force_exit.is_set():
                        print(
                            f"\r\nForce exit detected during save retry sleep. Aborting save."
                        )
                        break
                    time.sleep(1)

                continue  # retry after sleep

            # If save and load succeeded, break out of retry loop
            saved = True

        if saved:
            # Rename atomically if we got here
            try:
                for name in os.listdir(model_folder):
                    path = os.path.join(model_folder, name)

                    if not os.path.isfile(path):
                        continue

                    if model_type_extension in name and (not name.startswith(model_hash) or not name.endswith(
                        new_extension
                    )):  # ensure old model lingering new if exist also get removed, except new model
                        os.remove(path)

            except Exception:
                pass
            finally:
                os.rename(
                    model_metadata_path + model_extension + new_extension,
                    model_metadata_path + model_extension,
                )
                os.rename(
                    model_metadata_path + metadata_extension + new_extension,
                    model_metadata_path + metadata_extension,
                )
                print("Model saved.")
        else:
            print(f"\r\nSave aborted due to forced exit.")
            print(f"\r\nExiting.")
            sys.exit(0)  # Exit immediately
    finally:
        saving_in_progress.clear()
        if exit_after_save:
            print(f"\r\nExiting.")
            sys.exit(0)  # Exit immediately

def save_embedding(chunk_embeddings):
    saving_in_progress.set()
    saved = False
    try:
        np_embedding = np.asarray(chunk_embeddings)
        embedding_bytes = numpy.save({"embedding": np_embedding})
        embedding_hash = git_hash_bytes(embedding_bytes)
        embedding_type_extension = ".embedding"
        embedding_extension = ".safetensors"
        new_extension = ".new"
        embedding_path = model_folder + "/" + embedding_hash + embedding_type_extension

        while not saved:
            try:
                # Critical save section: do NOT abort mid-way
                with open(
                    embedding_path + embedding_extension + new_extension,
                    "wb",
                ) as f:
                    f.write(embedding_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                safe_open(
                    embedding_path + embedding_extension + new_extension,
                    framework="np",
                )

            except Exception as e:
                print("Save error:", e)
                # Check force_exit before retrying
                if force_exit.is_set():
                    print(
                        f"\r\nForce exit detected during save error handling. Aborting save."
                    )
                    break  # Abort save immediately

                # Sleep in 1 second chunks to be able to check force_exit regularly
                sleep_seconds = 300  # 5 minutes
                for _ in range(sleep_seconds):
                    if force_exit.is_set():
                        print(
                            f"\r\nForce exit detected during save retry sleep. Aborting save."
                        )
                        break
                    time.sleep(1)

                continue  # retry after sleep

            # If save and load succeeded, break out of retry loop
            saved = True

        if saved:
            # Rename atomically if we got here
            try:
                for name in os.listdir(model_folder):
                    path = os.path.join(model_folder, name)

                    if not os.path.isfile(path):
                        continue

                    if embedding_type_extension in name and (not name.startswith(embedding_hash) or not name.endswith(
                        new_extension
                    )):  # ensure old embedding lingering new if exist also get removed, except new embedding
                        os.remove(path)

            except Exception:
                pass
            finally:
                os.rename(
                    embedding_path + embedding_extension + new_extension,
                    embedding_path + embedding_extension,
                )
                print("embedding saved.")
        else:
            print(f"\r\nSave aborted due to forced exit.")
            print(f"\r\nExiting.")
            sys.exit(0)  # Exit immediately
    finally:
        saving_in_progress.clear()

class ModelFile:
    identifier: str
    ctime: int

    def __init__(self, identifier: str, ctime: int):
        self.identifier = identifier
        self.ctime = ctime


def check_model_file(folder, type = "retriever"):
    def latest_version(name):
        path = os.path.join(folder, name)
        with safe_open(path, framework="numpy") as f:
            identifier = name.split(".")[0]
            *_, ctime = os.stat(path)
            return ModelFile(identifier, ctime)

    safetensors_files = []
    for name in os.listdir(folder):
        if name.endswith("."+type+".safetensors"):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                model_file = latest_version(name)
                safetensors_files.append(model_file)

    return (
        None if not safetensors_files else max(safetensors_files, key=lambda x: x.ctime)
    )


latest_model_file: Optional[ModelFile] = None
model_file_extension = ".safetensors"
metadata_file_extension = ".metadata"

class GeneratorFunc:
    def unfold_1d(x, window_size, overlap, dim):
        """
        x: (..., T, ...)
        returns: (..., num_windows, size, ...)
        """
        T = x.shape[dim]
        # num_windows = (T - window_size) // overlap + 1
        # num_windows = 1 - overlap/(window_size - overlap)*math.floor((window_size-T)/overlap)
        # num_windows = (window_size - T) % overlap + T - overlap
        num_windows = math.ceil((T-overlap)/(window_size - overlap))

        # indices for windows
        base = mx.arange(window_size)                       # (size,)
        offsets = mx.arange(num_windows) * (window_size - overlap)      # (num_windows,)
        idx = offsets[:, None] + base[None, :]       # (num_windows, size)

        return mx.take(x, idx, axis=dim)

    # def to_samples(context_size, dataset):
    #     window_size = context_size + 1  # include target
    #     overlap = 2
    #     pad_len = (window_size - dataset.size % overlap) % overlap
    #     mx_dataset =  mx.array(dataset)
    #     mx_dataset = mx.pad(mx_dataset, ((0, pad_len)))
    #     frames = GeneratorFunc.unfold_1d(mx_dataset[None, :], window_size=window_size,
    #                         overlap=overlap, dim=-1).squeeze()
    #     return frames

    class DataSrc:
        def __init__(self, batch_size, input_len, s=0):
            self.batch_size = batch_size
            self.input_len = input_len
            # self.context_size = context_size
            # self.datasets = datasets
            # self.inputs = GeneratorFunc.to_samples(context_size, datasets)
            self.s = s

    def increment_s(self):
        self.s += self.batch_size
        if self.s >= self.input_len:
            self.s = 0

    def iterate_batches(self):
        while True:
            yield self.s
            self.increment_s()

def load_checkpoint(
    model, optimizer = None, model_file_identifier: Optional[str] = None, type="retriever"
):
    def _np2mx(np_dict: Dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
        new_list = []
        for k, v in np_dict.items():
            new_list.append((k, mx.array(v)))
        return new_list

    if model_file_identifier == None:
        print(f"model file not found.")
        return

    model_type_extension = ".retriever" if type=="retriever" else ".generator"
    target_file_path = model_folder + "/" + model_file_identifier + model_type_extension + model_file_extension

    target_metadata_file_path = (
        model_folder + "/" + model_file_identifier + model_type_extension + metadata_file_extension
    )

    if args.new_model or not os.path.isfile(target_file_path):
        if args.new_model:
            print(f"Starting with new model from scratch.")
        else:
            print(f"Checkpoint file {target_file_path} not found.")
    else:
        model_flattened_np = numpy.load_file(target_file_path)
        model_flattened = _np2mx(model_flattened_np)
        model_parameters = tree_unflatten(model_flattened)
        model.update(model_parameters)
        with open(target_metadata_file_path, "rb") as f:
            checkpoint = dill.load(f)
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer_state_flattened = _np2mx(checkpoint["optimizer"])
                optimizer_state = tree_unflatten(optimizer_state_flattened)
                optimizer.state = optimizer_state

            if "start" in checkpoint:
                return checkpoint["start"]

    return None

def load_embedding(
    embedding_file_identifier: Optional[str] = None
):
    if embedding_file_identifier == None:
        print(f"embedding file not found.")
        return

    type_extension = ".embedding"
    target_file_path = model_folder + "/" + embedding_file_identifier + type_extension + model_file_extension

    if args.new_model or not os.path.isfile(target_file_path):
        if args.new_model:
            print(f"Starting with new embedding from scratch.")
        else:
            print(f"Embedding file {target_file_path} not found.")
    else:
        embedding_np = numpy.load_file(target_file_path)
        return mx.array(embedding_np["embedding"])
    
    return None

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Callable[[Any], Any] = nn.relu
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads)
        self.ln1x = nn.LayerNorm(dims)
        self.ln1y = nn.LayerNorm(dims)
        self.ln2x = nn.LayerNorm(dims)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def __call__(self, x, y, cross = False):
        if cross:
            _x = self.ln1x(x)
            _y = self.ln1y(y)
            _x = self.attention(_x, _y, _y)
        else:
            _x = self.ln1x(x)
            _x = self.attention(_x, _x, _x)
        _x = self.dropout1(_x)
        x = x + _x

        _x = self.ln2x(x)
        _x = self.linear1(_x)
        _x = self.activation(_x)
        _x = self.dropout2(_x)
        _x = self.linear2(_x)
        x = x + _x
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation=nn.relu,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(
                dims, num_heads, mlp_dims, dropout, activation
            )
            for i in range(num_layers)
        ]
        self.ln = nn.LayerNorm(dims)
        self.checkpoint = checkpoint

    def __call__(self, x, y, cross = False):
        for l in self.layers:
            l = mx.checkpoint(l) if self.checkpoint else l
            x = l(x, y, cross)
        return self.ln(x)

class RecurringTransformerLM(nn.Module): # for query
    def __init__(
        self,
        vocab_size: int,
        window_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool,
    ):
        super().__init__()
        self.window_size = window_size
        self.dims = dims
        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = TransformerEncoder(
            num_layers, dims, num_heads, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, 64)

    def self_attention(self, x):
        x = self.embedding(x)
        x = x + self.pe(mx.arange(self.window_size))
        x = self.transformer(x, x)
        return x

    def self_attention_recurring(self, x):
        x = self.transformer(x, x)
        return x

    def cross_attention_recurring(self, x, y, n):
        y = self.embedding(y)
        y = y + self.pe(mx.arange(n*self.window_size, (n+1)*self.window_size, 1))
        x = self.transformer(x, y, True)
        return x

    def __call__(self, x):
        B = x.shape[0]
        L = x.shape[-1] # no batching for variable length query
        N = math.ceil(L/self.window_size)
        pad_len = N * self.window_size - L
        x = mx.pad(x, ((0, 0),(0, pad_len)))
        # N = 1: self
        # N = 2: self [-> cross -> self]
        # N = 3: self [-> cross -> self -> cross -> self]
        out = mx.zeros([B, self.window_size, self.dims])
        out = self.self_attention(x[:, :self.window_size, ...])
        for n in range(1, N):
            out = self.cross_attention_recurring(out, x[:, n*self.window_size:(n+1)*self.window_size, ...], n)
            out = self.self_attention_recurring(out)

        out = self.out_proj(out)
        return out.mean(axis=-2)

def to_indices(line):
    return np.array(spm_model.encode(line))
    
def process_chunk_embedding(retriever_model, tokenized_chunks, save_enabled = True):
    chunks_embedding_list = []
    state = [retriever_model.state]

    @partial(mx.compile, inputs=state)
    def step(tokenized_chunks_batch):
        return retriever_model(tokenized_chunks_batch)  # safe, fits in GPU

    for i in range(0, tokenized_chunks.shape[0], args.batch_size):
        print('processing chunk '+str(i)+' / '+str(tokenized_chunks.shape[0]), end='\r')
        tokenized_chunks_batch = tokenized_chunks[i:i+args.batch_size]
        chunks_embedding_batch = step(tokenized_chunks_batch)
        chunks_embedding_list.append(chunks_embedding_batch)
        mx.eval(chunks_embedding_batch)
    chunks_embedding = mx.concatenate(chunks_embedding_list, axis=0)
    if save_enabled:
        save_embedding(chunks_embedding)
    return chunks_embedding
    
def retriever_main(args):        
    def loss_fn(model, tokenized_question, tokenized_chunks_samples, p_total_draw, reduction="mean"):
        question_embedding = model(tokenized_question[None, :])
        chunks_embedding = model(tokenized_chunks_samples)
        dim = chunks_embedding.shape[-1]
        # question_embedding = question_embedding/mx.maximum(mx.linalg.norm(question_embedding, axis=-1), epsilon)
        # chunks_embedding = chunks_embedding/mx.maximum(mx.linalg.norm(chunks_embedding, axis=-1), epsilon)[:, None]
        logits = ((question_embedding @ chunks_embedding.T)/dim).squeeze()
        log_probs = mx.maximum(nn.log_softmax(logits), -18)
        # log_softmax = nn.log_softmax(logits)
        # log_probs_positive = mx.maximum(log_softmax[:p_total_draw], -18)
        # log_probs_negative = mx.maximum(log_softmax[p_total_draw:], -18)
        # y = mx.zeros(args.batch_size)
        # y[:p_total_draw] = 1/p_total_draw

        # return nn.losses.cross_entropy(logits, y, reduction=reduction)

        # loss = -log_probs_positive.mean()+log_probs_negative.mean()
        # return loss
        labels = mx.zeros(logits.shape[0])
        labels[:p_total_draw] = 1.0 / p_total_draw

        loss = -(labels * log_probs).mean()
        return loss
    
    def is_subarray(sub, arr):
        sub = np.asarray(sub)
        n = sub.size

        if n > arr.size:
            return False

        windows = np.lib.stride_tricks.sliding_window_view(arr, n)
        return np.any(np.all(windows == sub, axis=1))
    
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
        
    model = RecurringTransformerLM(
        args.vocab_size, args.context_size, args.num_blocks, args.dim, args.num_heads, args.checkpoint
    )

    model_file_identifier = args.identifier
    if model_file_identifier is None:
        latest_model_file = check_model_file(model_folder)
        if latest_model_file != None:
            model_file_identifier = latest_model_file.identifier
  

    start = load_checkpoint(model, optimizer, model_file_identifier)
    if start == None:
        start = 0
    
    mx.eval([model.parameters()])

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, *inputs)
        optimizer.update(model, grads)
        return loss
        
    chunks, files, questions, answers = datasets.load_retriever_dataset(args.context_size, save_dir=args.save_dir)
    np_tokenized_chunks = np.array([entry['tokenized_chunk'] for entry in chunks])
    tokenized_chunks = mx.array(np_tokenized_chunks)
    if not args.inference:

        losses = []
        tic = time.perf_counter()
        question_with_no_valid_chunks = set()
        for it in range(start, args.num_iters):
            i = it % len(questions)
            optimizer.learning_rate = min(1, it / args.lr_warmup) * args.learning_rate
            p_tokenized_chunk_ids = [chunk_id for file_id in questions[i]['file_ids'] for chunk_id in files[file_id]['chunk_ids'] if is_subarray(answers[i], np_tokenized_chunks[chunk_id])]        
            p_tokenized_chunk_ids_set = set(p_tokenized_chunk_ids)
            if len(p_tokenized_chunk_ids_set) == 0:
                question_with_no_valid_chunks.add(i)
                continue
            n_tokenized_chunk_ids = [x for x in range(len(chunks)) if x not in p_tokenized_chunk_ids_set]
            p_total_draw = min(len(p_tokenized_chunk_ids_set), args.batch_size // 2)
            n_total_draw = args.batch_size - p_total_draw
            p_tokenized_chunk_ids_random = random.sample(p_tokenized_chunk_ids, p_total_draw)
            n_tokenized_chunk_ids_random = random.sample(n_tokenized_chunk_ids, n_total_draw)
            tokenized_chunks_samples = mx.take_along_axis(tokenized_chunks, mx.array(np.array(p_tokenized_chunk_ids_random + n_tokenized_chunk_ids_random))[:, None], axis=0)
            tokenized_question = mx.array(questions[i]['tokenized'])
            loss = step([tokenized_question, tokenized_chunks_samples, p_total_draw])
            mx.eval(state)
            losses.append(loss.item())
            if (it + 1) % steps_per_report == 0:
                train_loss = sum(losses) / len(losses)
                toc = time.perf_counter()
                print(
                    f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                    f"It/sec {steps_per_report / (toc - tic):.3f}"
                )
                losses = []
                tic = time.perf_counter()

                if (it + 1) % steps_per_eval == 0 or finalise_and_exit.is_set():
                    save_invalid_question_ids(question_with_no_valid_chunks)
                    save(model, optimizer, it + 1, "retriever", finalise_and_exit.is_set())
                    # visualisation
                    if args.visualisation:
                        print(">>> "+spm_model.decode(questions[i]['tokenized']))
                        tokenized_question_preview = mx.array(np.array(questions[i]['tokenized']))
                        question_embedding = model(tokenized_question_preview[None, :])
                        chunks_embedding = process_chunk_embedding(model, tokenized_chunks, save = False)
                        # question_embedding = question_embedding/mx.maximum(mx.linalg.norm(question_embedding, axis=-1), epsilon)
                        # chunks_embedding = chunks_embedding/mx.maximum(mx.linalg.norm(chunks_embedding, axis=-1), epsilon)[:, None]
                        dim = chunks_embedding.shape[-1]
                        match = ((question_embedding @ chunks_embedding.T)/dim).squeeze()
                        top_k = mx.argpartition(-match, kth=args.top_k - 1)[: args.top_k]
                        top_k_list = top_k.tolist()
                        for j in top_k_list:
                            decoded = spm_model.decode(chunks[j]['tokenized_chunk'].tolist())
                            print(files[chunks[j]['file_id']]['file_name']+" - "+decoded)
    else:
        while True:
            user_input = input(">>> ")
            if user_input == "quit":
                print("bye bye!")
                break
            user_input_indices = to_indices(user_input.strip().lower())
            tokenized_question = mx.array(np.array(user_input_indices))
            question_embedding = model(tokenized_question[None, :])
            # chunks_embedding = model(tokenized_chunks)
            # to do list: Once the retriever model is finalized, save chunk embeddings to avoid redundant computation.
            chunks_embedding_list = []
            for i in range(0, tokenized_chunks.shape[0], args.batch_size):
                print('processing chunk '+str(i)+' / '+str(tokenized_chunks.shape[0]), end='\r')
                tokenized_chunks_batch = tokenized_chunks[i:i+args.batch_size]
                chunks_embedding_batch = model(tokenized_chunks_batch)  # safe, fits in GPU
                chunks_embedding_list.append(chunks_embedding_batch)
                mx.eval(chunks_embedding_batch)
            chunks_embedding = mx.concatenate(chunks_embedding_list, axis=0)
            dim = chunks_embedding.shape[-1]
            match = ((question_embedding @ chunks_embedding.T)/dim).squeeze()
            top_k = mx.argpartition(-match, kth=args.top_k - 1)[: args.top_k]
            top_k_list = np.asarray(top_k)
            for j in top_k_list:
                decoded = spm_model.decode(chunks[j]['tokenized_chunk'].tolist())
                print(files[chunks[j]['file_id']]['file_name']+" - "+decoded)


def main(args):
    class TransformerLMEncoder(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            dims: int,
            num_heads: int,
            checkpoint: bool
        ):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, dims)
            self.pe = nn.SinusoidalPositionalEncoding(dims)
            self.transformer = nn.TransformerEncoder(
                num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
            )
            self.out_proj = nn.Linear(dims, vocab_size)
            self.type_embedding = nn.Embedding(2, dims) # query, doc

        def __call__(self, x, type = 0):
            L = x.shape[1]
            x = self.embedding(x)
            x = x + self.type_embedding(type)
            x = x + self.pe(mx.arange(L))
            x = self.transformer(x, None)
            return x
            
    class TransformerLMDecoder(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            dims: int,
            num_heads: int,
            checkpoint: bool
        ):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, dims)
            self.pe = nn.SinusoidalPositionalEncoding(dims)
            self.transformer = nn.TransformerEncoder(
                num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
            )

        def __call__(self, x):
            L = x.shape[1]
            x = self.embedding(x)
            x = x + self.pe(mx.arange(L))
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            x = self.transformer(x, mask)
            return x
        
    class EncoderDecoderLM(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            dims: int,
            num_heads: int,
            checkpoint: bool
        ):
            super().__init__()
            self.encoder = TransformerLMEncoder(vocab_size, num_layers, dims, num_heads, checkpoint)
            self.referer = TransformerEncoder(1, dims, num_heads, checkpoint=checkpoint)
            self.decoder = TransformerLMDecoder(vocab_size, num_layers, dims, num_heads, checkpoint)
            self.out_proj = nn.Linear(dims, vocab_size)


        def encode(self, qs, chunks):
            B, N, T = chunks.shape
            chunks_flattened = chunks.reshape((B*N, T))
            qs_encoded = self.encoder(qs, 0)
            chunks_flattened_encoded = self.encoder(chunks_flattened, 1)
            chunks_encoded  = chunks_flattened_encoded.reshape((B, N, T, chunks_flattened_encoded.shape[-1]))
            return qs_encoded, chunks_encoded
        
        def decode(self, qs_encoded, chunks_encoded, ans):
            _, N, *_ = chunks_encoded.shape
            ans_decoded = self.decoder(ans)
            ans_decoded = self.referer(ans_decoded, qs_encoded, True)
            for i in range(N):
                ans_decoded = self.referer(ans_decoded, chunks_encoded[:, i, ...], True)
            return self.out_proj(ans_decoded)
        
        def __call__(self):
            pass


    # Load vocab and dataset:
    qas = datasets.load_generator_dataset(
        save_dir=args.save_dir
    )

    unsupported_qa_path = os.path.join(args.save_dir+"/qa","unsupported-wikipedia-train.json")
    if os.path.exists(unsupported_qa_path):
        with open(unsupported_qa_path, "r", encoding="utf-8") as input_file:
            qa = json.load(input_file)
            mask = np.ones(len(qas), dtype=bool)
            mask[qa["question_ids"]] = False
            qas = qas[mask]


    qas_chunks = mx.zeros((qas.shape[0], args.top_k, args.context_size), dtype=mx.int32)

    # load pretrained retriever
    retriever_model = RecurringTransformerLM(
        args.vocab_size, args.context_size, args.num_blocks, args.dim, args.num_heads, args.checkpoint
    )

    retriever_model_file_identifier = args.retriever_identifier
    if retriever_model_file_identifier is None:
        latest_retriever_model_file = check_model_file(model_folder, "retriever")
        if latest_retriever_model_file != None:
            retriever_model_file_identifier = latest_retriever_model_file.identifier

    embedding_file_identifier = args.embedding_identifier
    if embedding_file_identifier is None:
        latest_embedding_file = check_model_file(model_folder, "embedding")
        if latest_embedding_file != None:
            embedding_file_identifier = latest_embedding_file.identifier
  
    _ = load_checkpoint(retriever_model, None, retriever_model_file_identifier)
    chunks_embedding = load_embedding(embedding_file_identifier)
    chunks, *_ = datasets.load_retriever_dataset(args.context_size, save_dir=args.save_dir)
    np_tokenized_chunks = np.array([entry['tokenized_chunk'] for entry in chunks])
    tokenized_chunks = mx.array(np_tokenized_chunks)
    
    if chunks_embedding is None:
        chunks_embedding = process_chunk_embedding(retriever_model, tokenized_chunks)

    state = [retriever_model.state]

    @partial(mx.compile, inputs=state)
    def qa_step(tokenized_question, chunks_embedding):
        question_embedding = retriever_model(tokenized_question)  # safe, fits in GPU
        dim = chunks_embedding.shape[-1]
        match = ((question_embedding @ chunks_embedding.T)/dim).squeeze()
        top_k = mx.argpartition(-match, kth=args.top_k - 1)[: args.top_k]
        return mx.take(tokenized_chunks, top_k, axis=0)
    
    for i, qa in enumerate(qas):
        print('processing qa pair '+str(i)+' / '+str(qas.shape[0]), end='\r')
        tokenized_question = mx.array(qa[0])
        current_qa_chunks = qa_step(tokenized_question[None, :], chunks_embedding)
        mx.eval(current_qa_chunks)
        qas_chunks[i, ...] = current_qa_chunks

    # Initialize model:
    model = EncoderDecoderLM(
        args.vocab_size, args.num_blocks, args.dim, args.num_heads, args.checkpoint
    )

    model_file_identifier = args.identifier
    if model_file_identifier is None:
        latest_model_file = check_model_file(model_folder, "generator")
        if latest_model_file != None:
            model_file_identifier = latest_model_file.identifier

    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Total parameters: {nparams / 1024**2:.3f} M")

    def loss_fn(model, inputs, reduction="mean"):
        qas, chunks, selected_indices = inputs
        qas_batch = mx.take(qas, selected_indices, axis=0)
        chunks_batch = mx.take(chunks, selected_indices, axis=0)
        ans_batch = qas_batch[:,1]
        qs_encoded, chunks_encoded = model.encode(qas_batch[:,0], chunks_batch)
        logits = model.decode(qs_encoded, chunks_encoded, ans_batch)
        y = mx.zeros_like(qas_batch[:,1])
        y[..., :-1] = ans_batch[..., 1:]
        return nn.losses.cross_entropy(logits, y, reduction=reduction)

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )

    start = load_checkpoint(model, optimizer, model_file_identifier, "generator")
    if start == None:
        start = 0
    train_range = range(start, args.num_iters)
    mx.eval(model.parameters())

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs)
        optimizer.update(model, grads)
        return loss

    if not args.inference:
        # train_iterator = data_src.iterate_batches()
        losses = []
        tic = time.perf_counter()
        qas = mx.array(qas)
        qas_indices = range(qas.shape[0])
        for it in train_range:
            optimizer.learning_rate = min(1, it / args.lr_warmup) * args.learning_rate
            sampled_indices = random.sample(qas_indices, k=args.batch_size)
            loss = step([qas, qas_chunks, mx.array(sampled_indices, dtype=mx.int32)])
            mx.eval(state)
            losses.append(loss.item())
            if (it + 1) % steps_per_report == 0:
                train_loss = sum(losses) / len(losses)
                toc = time.perf_counter()
                print(
                    f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                    f"It/sec {steps_per_report / (toc - tic):.3f}"
                )
                losses = []
                tic = time.perf_counter()

            if (it + 1) % steps_per_eval == 0 or finalise_and_exit.is_set():
                save(model, optimizer, it + 1, "generator", finalise_and_exit.is_set())

                # visualisation
                if args.visualisation:
                    output_deque = deque(maxlen=args.context_size)
                    output_deque.append(BOS)
                    user_input = "from which country did angola achieve independence in 1975?"
                    print(">>> " + user_input)
                    query_indices = to_indices(user_input.strip().lower())
                    query = mx.array(np.array(query_indices))
                    query_embedding = retriever_model(query[None, :])
                    dim = chunks_embedding.shape[-1]
                    match = ((query_embedding @ chunks_embedding.T)/dim).squeeze()
                    top_k = mx.argpartition(-match, kth=args.top_k - 1)[: args.top_k]
                    chunks = mx.take(tokenized_chunks, top_k, axis=0)
                    qs_encoded, chunks_encoded = model.encode(query[None, :], chunks[None, :])
                    initial = True
                    for _ in range(100):  # visualise only first 100 tokens or less

                        output = mx.array(np.array(output_deque))[None, :]
                        logits = model.decode(qs_encoded, chunks_encoded, output)

                        token = logits[0, -1, :]
                        top_k = mx.argpartition(-token, kth=args.top_k - 1)[: args.top_k]
                        token = token[top_k]
                        output_logits = nn.log_softmax(token / args.temperature)
                        index = mx.random.categorical(output_logits)
                        detached_index = index.item()
                        top_k_detached_index = top_k[detached_index].item()
                        if top_k_detached_index == EOS:
                            break
                        output_deque.append(top_k_detached_index)
                        piece = spm_model.id_to_piece(top_k_detached_index)
                        if piece[0] == '▁':
                            if initial:
                                print(piece[1:], end="")
                            else:
                                print(" "+piece[1:], end="")
                        else:
                            print(piece, end="")
                        if initial:
                            initial = False
    else:
        while True:
            output_deque = deque(maxlen=args.context_size)
            output_deque.append(BOS)
            user_input = input(">>> ")
            if user_input == "quit":
                print("bye bye!")
                break
            query_indices = to_indices(user_input.strip().lower())
            query = mx.array(np.array(query_indices))
            query_embedding = retriever_model(query[None, :])
            dim = chunks_embedding.shape[-1]
            match = ((query_embedding @ chunks_embedding.T)/dim).squeeze()
            top_k = mx.argpartition(-match, kth=args.top_k - 1)[: args.top_k]
            chunks = mx.take(tokenized_chunks, top_k, axis=0)
            qs_encoded, chunks_encoded = model.encode(query[None, :], chunks[None, :])
            initial = True
            while True:
                output = mx.array(np.array(output_deque))[None, :]
                logits = model.decode(qs_encoded, chunks_encoded, output)

                token = logits[0, -1, :]
                top_k = mx.argpartition(-token, kth=args.top_k - 1)[: args.top_k]
                token = token[top_k]
                output_logits = nn.log_softmax(token / args.temperature)
                index = mx.random.categorical(output_logits)
                detached_index = index.item()
                top_k_detached_index = top_k[detached_index].item()
                if top_k_detached_index == EOS:
                    break
                output_deque.append(top_k_detached_index)
                piece = spm_model.id_to_piece(top_k_detached_index)
                if piece[0] == '▁':
                    if initial:
                        print(piece[1:], end="")
                    else:
                        print(" "+piece[1:], end="")
                else:
                    print(piece, end="")
                if initial:
                    initial = False
            print("")

def dataset_main(args):
    datasets.prepare_generator_qa_dataset(args.from_dir, args.to_dir)
    datasets.qa_json_to_txt(args.save_dir)
    datasets.to_lowercase_files(args.from_dir, args.to_dir)
    datasets.train_tokenizer(args.save_dir)

if __name__ == "__main__":
    if args.target == "dataset":
        dataset_main(args)
    elif args.target == "retriever":
        retriever_main(args)
    elif args.target == "all":
        main(args)
