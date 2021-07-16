import math
import random
import time
from pathlib import Path

import dill
import numpy as np
import sentencepiece as spm
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from model import *
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import BucketIterator, Field
from torchtext.legacy.datasets import Multi30k, TranslationDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from translate import *
from utils import *


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    sp.load("pretrained/de.model")
    return sp.encode_as_pieces(text)


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    sp.load("pretrained/en.model")
    return sp.encode_as_pieces(text)


with open("pretrained/SRC.Field", "rb") as f:
    SRC = dill.load(f)

with open("pretrained/TRG.Field", "rb") as f:
    TRG = dill.load(f)

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

    enc = Encoder(
        INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device
    )

    dec = Decoder(
        OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device
    )
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = T5ForConditionalGeneration.from_pretrained(
        "pretrained/T5", return_dict=True
    )
    tokenizer = T5Tokenizer.from_pretrained("pretrained/T5")
    attn_model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    return model, tokenizer, attn_model


model, tokenizer, attn_model = load_nlp()


def translate_sentence_t5(input_sentence):

    input_ids = tokenizer(input_sentence, return_tensors="pt").input_ids

    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=150,
        num_beams=5,
        temperature=0.1,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )
    preds = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in generated_ids
    ]
    return preds[0]


###########################################

st.title("German to English Translation")

text = st.text_area("Text to be translated")

sp = spm.SentencePieceProcessor()

sp.load("pretrained/en.model")

if text:
    translated = translate_sentence_t5(text)
    st.markdown("## Translated Sentence - T5")
    st.write(translated)
    translated = translate_sentence(translated, SRC, TRG, attn_model, "cpu")[0]
    st.markdown("## Translated Sentence - Transformers")
    st.write(sp.decode(translated))
