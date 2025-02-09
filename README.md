# CFMAD

The implement of our paper "[Counterfactual Debating with Preset Stances for Hallucination Elimination of LLMs](https://arxiv.org/abs/2406.11514)" (COLING 2025).

## Introduction

- We propose to preset various stances for LLMs, overriding their inherent biases and beliefs to address the overconfidence issue of LLMs.
- We propose a CFMAD framework, which instructs LLMs to generate abduction with preset stances and then conduct counterfactual debate to eliminate incorrect answers.

## Dataset

This repository uses data from the [Hover](https://hover-nlp.github.io/), [BoolQ](https://github.com/google-research-datasets/boolean-questions), [CosmosQA](https://wilburone.github.io/cosmos/) and [CommenseQA](https://www.tau-nlp.org/commonsenseqa) datasets.

## Setup

Obtain an OpenAI API key and save it to the environment variable `OPENAI_API_KEY`.

## Citation

```bibtex
@inproceedings{cfmad,
  author       = {Yi Fang and
                  Moxin Li and
                  Wenjie Wang and
                  Lin Hui and
                  Fuli Feng},
  title        = {Counterfactual Debating with Preset Stances for Hallucination Elimination
                  of LLMs},
  booktitle    = {{COLING}},
  pages        = {10554--10568},
  publisher    = {Association for Computational Linguistics},
  year         = {2025}
}
```
