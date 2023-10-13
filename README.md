# SkimLit
## Overview

This project aims to replicate and extend the work presented in the research paper titled "PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts." The primary goal is to develop a machine learning model that can classify sentences in a research paper abstract into five different categories: Background, Objective, Method, Result, and Conclusion.

## Hybrid Model

To achieve this classification task, we have implemented a hybrid model that combines various sub-models, including:

- Char-Level Tokenized Model
- Word-Level Tokenized Model
- Line Number Model
- Total Lines Model

These sub-models are designed to capture different aspects of the abstract text, including character-level patterns, word-level semantics, and structural features.

## Training and Replication

Our approach involves creating and training individual models for each subtask. We then integrate these models into a single model, which we fine-tune on the dataset to replicate the results of the original research paper.

## Code

All the code, including the individual models and the hybrid model, can be found in the provided Jupyter Notebook file.

## Dataset

We use the "PubMed 200k RCT" dataset, which is a valuable resource for sequential sentence classification in medical abstracts.

## Acknowledgments

This project is inspired by and replicates the work of the original authors of the research paper. The goal is to contribute to the field of text classification and encourage further research in this domain.

Feel free to explore the code and documentation to understand how the models are implemented and trained to classify sentences in research paper abstracts.

## Link
[PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071)
[Link to Original dataset](https://github.com/Franck-Dernoncourt/pubmed-rct)
