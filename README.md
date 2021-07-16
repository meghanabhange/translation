# Paying Attention to The Translations ✨

## Demo

PLACEHOLDER

## How To Run The Demo Locally

*Installations*

```
pip install requirements.py
```

*Getting Data*

The trained models are available as GitHub Releases with the repository. 

```
python -m download OUTPUT_DIR
```

*Running App*

``
streamlit run seq2seq/app.py
``

## Evaluation 

```
pip install requirements.py
python -m download OUTPUT_DIR
python -m predictandevaluate OUTPUT_DIR
```
---

# Trained Models 

## [Sentencepiece Tokenizer](https://github.com/meghanabhange/translation/releases/tag/0.2)
Trained on [stanford NMT English German translation](https://nlp.stanford.edu/projects/nmt/) dataset with vocabulary size 32k

## [T5 Huggingface Model](https://github.com/meghanabhange/translation/releases/tag/0.1) 


# Notebooks 

- 1. Experiment 1. Transformer Seq2Seq model using attention 
- 2. Experiment 2. T5 translation model

# Metrics 

## 1. Experiment 1. Transformer Seq2Seq model using attention 

- BLEU : 
- BERTScore : 
- WER (Word Error Rate) : 

## 2. Experiment 2. T5 translation model

- BLEU : 
- BERTScore : 
- WER (Word Error Rate) : 
