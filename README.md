# Awesome-Privacy-Preserving-LLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
LLMs have taken the world by storm, showing outstanding capabilities in several NLP-related domains. They have been proven to have astonishing emergent capabilities and unfortunately it has become painfully obvious that memorization is one of them. While this is not a problem for models dealing with public data, when the task at hand requires to deal with sensitive data this issue cannot be overlooked. This is why, spurring from our research survey, we present here a curated list of papers on the subjects of LLMs data memorization, the privacy attacks that this allows and potential solutions, including data anonymization, Differential Privacy and Machine Unlearning.

## Table of Contents

- [Awesome-Privacy-Preserving-LLMs](#awesome-privacy-preserving-llms)
  - [Attacks](#attacks)
  - [Data anonymization](#data-anonymization)
  - [Data anonymization with Differential Privacy](#data-anonymization-with-differential-privacy)
  - [Pre-training with Differential Privacy](#pretraining-with-differential-privacy)
  - [Fine-tuning with Differential Privacy](#fine-tuning-with-differential-privacy)
  - [Parameter-Efficient Fine-Tuning with Differential Privacy](#parameter-efficient-fine-tuning-with-differential-privacy)
  - [Reinforcement Learning with Differential Privacy](#reinforcement-learning-with-differential-privacy)
  - [Inference with Differential Privacy](inference-with-differential-privacy)
  - [Federated Learning with Differential Privacy](federated-learning-with-differential-privacy)
  - [Machine Unlearning](machine-unlearning)
  - [Tools and Frameworks](tools-and-frameworks)


## Attacks






## Data anonymization

[Guaranteeing anonymity when sharing medical data, the Datafly System](https://pubmed.ncbi.nlm.nih.gov/9357587/#:~:text=We%20present%20a%20computer%20program,details%20found%20within%20the%20data.) - Foundational paper for k-anonymity
[Automated anonymization of text documents](https://www.semanticscholar.org/paper/Automated-anonymization-of-text-documents-Mamede-Baptista/3c31662b2f81ad149c8e04025ad148b496d5118e) - Modular anonymization system for text documents.
[Data Anonymization for Pervasive Health Care: Systematic Literature Mapping Study](https://pubmed.ncbi.nlm.nih.gov/34652278/) - Examines methods for text perturbation (like microaggregation or data swapping).
[DataSifterText: Partially Synthetic Text Generation for Sensitive Clinical Notes](https://link.springer.com/article/10.1007/s10916-022-01880-6) - Use BERT to impute previously masked sensitive info.
[Natural Text Anonymization Using Universal Transformer with a Self-attention](https://www.semanticscholar.org/paper/Natural-Text-Anonymization-Using-Universal-with-a-Romanov-Fedotova/acb2dfc52cb13762778f20a51c32071762afd867) - Text anonymization system that uses a universal transformer model to generate anonymized text incorporating filtered and smoothed features from the original text.
[Deep Reinforcement Learning-based Text Anonymization against Private-Attribute Inference](https://aclanthology.org/D19-1240/) - Text anonymization system that manipulates the embeddings with an RL-based privacy-preserver.
[Recovering from Privacy-Preserving Masking with Large Language Models](https://arxiv.org/abs/2309.08628) - Use an LLM to impute previously masked tokens.
[Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection](https://arxiv.org/abs/2309.03057) - System using two local LLMs for anonymization and de-anonymization and in the middle use a black-box LLM.


## Data anonymization with Differential Privacy

[Broadening the Scope of Differential Privacy Using Metrics](https://link.springer.com/chapter/10.1007/978-3-642-39077-7_5) - Foundational paper for Metric Differential Privacy
[ADePT: Auto-encoder based Differentially Private Text Transformation](https://arxiv.org/abs/2102.01502) - Auto-encoder based DP algorithm to anonymize text while retaining utility. 
[When differential privacy meets NLP: The devil is in the detail](https://arxiv.org/abs/2109.03175) - Formal analysis of ADePT, highlights some issues with privacy guarantees.
[DP-VAE: Human-Readable Text Anonymization for Online Reviews with Differentially Private Variational Autoencoders](https://dl.acm.org/doi/abs/10.1145/3485447.3512232) - End-to-end DP-VAE for text anonymization.
[DP-BART for Privatized Text Rewriting under Local Differential Privacy](https://arxiv.org/abs/2302.07636) - Text privatization model based on BART.
[Sanitizing Sentence Embeddings (and Labels) for Local Differential Privacy](https://dl.acm.org/doi/abs/10.1145/3543507.3583512) - Sanitization system based on Purkayastha Mechanism.
[Differential Privacy for Text Analytics via Natural Text Sanitization](https://arxiv.org/abs/2106.01221) - They propose SANTEXT to replace sensitive tokens.
[A Customized Text Sanitization Mechanism with Differential Privacy](https://arxiv.org/abs/2207.01193) - They propose CUSTEXT to replace sensitive tokens.
[InferDPT: Privacy-Preserving Inference for Black-box Large Language Model](https://cs.paperswithcode.com/paper/privinfer-privacy-preserving-inference-for) - They propose RANTEXT to replace sensitive tokens.
[Leveraging Hierarchical Representations for Preserving Privacy and Utility in Text](https://arxiv.org/abs/1910.08917) - New noise distribution specifically devised for Metric DP.
[A Differentially Private Text Perturbation Method Using a Regularized Mahalanobis Metric](https://arxiv.org/abs/2010.11947) - Regularized Mahalanobis Metric for text perturbation.
[On a Utilitarian Approach to Privacy Preserving Text Generation](https://arxiv.org/abs/2104.11838) - Based on Vickrey auction, they balance the choice beetween first and second neighbours using a tuning parameter.
[Guiding Text-to-Text Privatization by Syntax](https://arxiv.org/abs/2306.01471) - Includes grammatical categories into the privatization process to preserve syntax.
[The Limits of Word Level Differential Privacy](https://arxiv.org/abs/2205.02130) - Paraphrasing model obtained by fine-tuning GPT2.

## Pre-training with Differential Privacy





## Fine-tuning with Differential Privacy






## Parameter-Efficient Fine-Tuning with Differential Privacy







## Reinforcement Learning with Differential Privacy



## Inference with Differential Privacy




## Federated Learning with Differential Privacy



## Machine Unlearning




## Tools and Frameworks
