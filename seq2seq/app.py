from pathlib import Path
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import math
import random
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import torchtext
from torchtext.legacy.data import BucketIterator, Field
from torchtext.legacy.datasets import Multi30k, TranslationDataset

from model import *
from utils import *
from translate import *


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    # sp.load("de.model")
    # return sp.encode_as_pieces(text)
    return text.split()


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    # sp.load("en.model")
    # return sp.encode_as_pieces(text)
    return text.split()


SRC = Field(
    tokenize=tokenize_de,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    truncate_first=True,
    fix_length=64,
    batch_first=True,
)

TRG = Field(
    tokenize=tokenize_en,
    init_token="<sos>",
    eos_token="<eos>",
    fix_length=64,
    lower=True,
    truncate_first=True,
    batch_first=True,
)

train_data = TranslationDataset(
    path='data/train_sample', exts=('.en', '.de'),
    fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

device = "cpu"

@st.cache(allow_output_mutation=True)
def load_nlp():
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.05
    DEC_DROPOUT = 0.05

    enc = Encoder(INPUT_DIM, 
                  HID_DIM, 
                  ENC_LAYERS, 
                  ENC_HEADS, 
                  ENC_PF_DIM, 
                  ENC_DROPOUT, 
                  device)

    dec = Decoder(OUTPUT_DIM, 
                  HID_DIM, 
                  DEC_LAYERS, 
                  DEC_HEADS, 
                  DEC_PF_DIM, 
                  DEC_DROPOUT, 
                  device)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = T5ForConditionalGeneration.from_pretrained("pretrained/T5", return_dict=True)
    tokenizer = T5Tokenizer.from_pretrained("pretrained/T5")
    attn_model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    return model, tokenizer, attn_model

model, tokenizer, attn_model = load_nlp()

def translate_sentence_t5(input_sentence):
  
  input_ids = tokenizer(input_sentence, return_tensors='pt').input_ids

  generated_ids = model.generate(
                input_ids = input_ids,
                max_length=150, 
                num_beams=5,
                temperature =0.1,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
  return preds[0]

###########################################

st.title("German to English Translation")

text = st.text_area("Text to be translated")


if text:
    translated = translate_sentence_t5(text)
    st.markdown("## Translated Sentence - T5")
    st.write(translated)
    translated = translate_sentence(translated, SRC, TRG, attn_model, "cpu")[0]
    st.markdown("## Translated Sentence - Transformers")
    st.write(" ".join(translated))