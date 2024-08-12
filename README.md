![Taxonomy of Privacy-Preserving LLMs](https://github.com/michele17284/Awesome-Privacy-Preserving-LLMs/blob/main/Taxonomy.png?raw=true)

# Awesome-Privacy-Preserving-LLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
LLMs have taken the world by storm, showing outstanding capabilities in several NLP-related domains. They have been proven to have astonishing emergent capabilities and unfortunately it has become painfully obvious that memorization is one of them. While this is not a problem for models dealing with public data, when the task at hand requires to deal with sensitive data this issue cannot be overlooked. This is why, spurring from our [research survey](https://arxiv.org/abs/2408.05212), we present here a curated list of papers on the subjects of LLMs data memorization, the privacy attacks that this allows and potential solutions, including data anonymization, Differential Privacy and Machine Unlearning.

## Table of Contents

- [Awesome-Privacy-Preserving-LLMs](#awesome-privacy-preserving-llms)
  - [Data Extraction](#data-extraction)
  - [Membership Inference Attacks](#membership-inference-attacks)
  - [Model Inversion](model-inversion)
  - [Re-Identification from Anonymized Data](#re-identification-from-anonymized-data)
  - [Attacks against Synthetic Data Generators](#attacks-against-synthetic-data-generators)
  - [Data anonymization](#data-anonymization)
  - [Data anonymization with Differential Privacy](#data-anonymization-with-differential-privacy)
  - [Pre-training with Differential Privacy](#pretraining-with-differential-privacy)
  - [Fine-tuning with Differential Privacy](#fine-tuning-with-differential-privacy)
  - [Parameter-Efficient Fine-Tuning with Differential Privacy](#parameter-efficient-fine-tuning-with-differential-privacy)
  - [Reinforcement Learning with Differential Privacy](#reinforcement-learning-with-differential-privacy)
  - [Inference with Differential Privacy](#inference-with-differential-privacy)
  - [Federated Learning with Differential Privacy](#federated-learning-with-differential-privacy)
  - [Machine Unlearning](#machine-unlearning)
  - [Tools and Frameworks](#tools-and-frameworks)


## Data Extraction

- [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646) Shows that it is possible to reconstruct training data with black-box access.
- [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) Query GPT-2 to extract training data.
- [Are Large Pre-Trained Language Models Leaking Your Personal Information?](https://arxiv.org/abs/2205.12628) Query LMs for email addresses and names, finding that the models are prone to leaking.
- [Scalable extraction of training data from (production) language models](https://arxiv.org/abs/2311.17035) Studies data extraction without prior knowledge about the dataset.
- [DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models](https://arxiv.org/abs/2306.11698) Explore the leakage of training data of GPT models.
- [Analyzing Leakage of Personally Identifiable Information in Language Models](https://arxiv.org/abs/2302.00539) Propose to solve a masked language modeling task to reconstruct masked personal information from a sentence.
- [Dataset reconstruction attack against language models](https://www.researchgate.net/publication/353191026_Dataset_Reconstruction_Attack_against_Language_Models) Reconstruct the training data of a finetuned GPT-2.
- [Scalable extraction of training data from (production) language models](https://arxiv.org/abs/2311.17035) Querying open-source models they were able to verify the success of Carlini's attack procedure by accessing to the training data only to verify the attack.
- [Quantifying Association Capabilities of Large Language Models and Its Implications on Privacy Leakage](https://arxiv.org/abs/2305.12707) Shows that it is possible to recover 3\% of the training emails from a 20 billion parameter model with attacks that require association.
- [ETHICIST: Targeted Training Data Extraction Through Loss Smoothed Soft Prompting and Calibrated Confidence Estimation](https://aclanthology.org/2023.acl-long.709/) Propose an attack to recover a certain suffix given a precise prefix, known to be in the pre-training data.
- [Controlling the Extraction of Memorized Data from Large Language Models via Prompt-Tuning](https://arxiv.org/abs/2305.11759) Similar approach to ETHICIST but based on soft propmpt tuning.
- [ProPILE: Probing Privacy Leakage in Large Language Models](https://arxiv.org/abs/2307.01881) Tool designed to test models with black-box and white-box attacks against their tendency to release PII.
- [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527) The original goal of a prompt is changed with malicious text.
- ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825) Characterize a large set of jailbreak prompts and evaluate their effectiveness against different LLMs.
- [ChatGPT_DAN](https://github.com/0xk1h0/ChatGPT_DAN) ChatGPT jailbreak that makes it behave like a fictional assistant.
- [Multi-step Jailbreaking Privacy Attacks on ChatGPT](https://arxiv.org/abs/2304.05197) Propose a multi-step jailbreak prompt to extract personal information, based on Chain-of-Thought.

## Membership Inference Attacks

- [Membership inference attack susceptibility of clinical language models]([https://arxiv.org/abs/2211.09527](https://arxiv.org/abs/2104.08305) Conduct attacks against BERT and GPT-2 trained on clinical data and show that DP helps mitigate privacy leakage.
- [Auditing Data Provenance in Text-Generation Models](https://arxiv.org/abs/1811.00513) Attack exploits the tendency of LMs to rank rare words higher when they are in the same context in which they were seen during training.
- [Membership Inference Attacks on Sequence-to-Sequence Models: Is My Data In Your Machine Translation System?](https://arxiv.org/abs/1904.05506) Train an attack classifier only on features that can be extracted from the output sequences.
- [Membership inference attacks from first principles](https://arxiv.org/abs/2112.03570) Train numerous shadow GPT-2 models to measure the probability of observing a certain likelihood of an example in models trained and not trained on it.
- [Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789) Design attacks that threshold the average log likelihood of the top k rarest words to ascertain whether or not the example is part of the training data.
- [Quantifying Privacy Risks of Masked Language Models Using Membership Inference Attacks](https://aclanthology.org/2022.emnlp-main.570/) Demonstrate the effectiveness of MIA designed against LMs trained with Masked Language Modeling objectives.
- [Membership Inference Attacks against Language Models via Neighbourhood Comparison](https://arxiv.org/abs/2305.18462) Use a sentence and its perturbed version and propose that the model should act similarly with both if the sentence is not in the training set.
- [On the privacy risk of in-context learning](https://virtual2023.aclweb.org/paper_TrustNLP_13.html) Show that LLMs, are vulnerable to MIAs that target datasets used during prompt training.

## Model Inversion

- [Privacy Risks of General-Purpose Language Models](https://www.semanticscholar.org/paper/Privacy-Risks-of-General-Purpose-Language-Models-Pan-Zhang/b3c73de96640ee858f83c3f0eda2a3d15d59b847) Design the first model inversion attack against Transformer-based Language Models.
- [Information Leakage in Embedding Models](https://arxiv.org/abs/2004.00053) Attack model is a NN trained with multiset prediction loss so that it is possible for the model to predict each word in the embedding also conditioned on the words already predicted.
- [Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence](https://arxiv.org/abs/2305.03010) Implement the inversion as a generative LM task with a decoder-only model conditioned on the embedding of the sentence to invert as the first token representation.
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) Propose an iterative method to reconstruct the input text of models that produce embeddings for documents generating iteratively different hyphothesis that may justify the observed embedding.
- [Privacy Leakage in Text Classification A Data Extraction Approach](https://aclanthology.org/2022.privatenlp-1.3/) Inject canaries in the training data and then reconstruct a partially masked sentence to search for tokens that maximize the probability of the target label.
- [Canary Extraction in Natural Language Understanding Models](https://arxiv.org/abs/2203.13920) Very similar to the previous work but uses a different reconstruction method.
- [Text Revealer: Private Text Reconstruction via Model Inversion Attacks against Transformers](https://arxiv.org/abs/2209.10505) Create a dataset mimicking unknown training data, train a model on it, and then adjust it by perturbing word embeddings to reduce classification loss on a target model.
- [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935) Demonstrates that if the gradients are openly accessible, it is possible to reconstruct the training data.
- [TAG: Gradient Attack on Transformer-based Language Models](https://arxiv.org/abs/2103.06819) Recovers up to 50% of the original tokens attacking BERT.
- [LAMP: Extracting Text from Gradients with Language Model Priors](https://arxiv.org/abs/2202.08827) Simultaneously train the attack model to minimize the difference between reconstruction gradients and choosing at each iteration only sequences that have low perplexity according to an external LM.
- [Recovering Private Text in Federated Learning of Language Models](https://arxiv.org/abs/2205.08514) Recover from the gradients a bag of words for the sentence to extract and then perform beam search to effectively reconstruct the sentence.

## Re-Identification from Anonymized Data

- [Robust de-anonymization of large sparse datasets](https://arxiv.org/abs/cs/0610105) Show that an attacker can use background knowledge or external data to reconstruct the identity of a user in a sparse dataset describing users preferences or transactions.
- [Estimating the success of re-identifications in incomplete datasets using generative models](https://www.nature.com/articles/s41467-019-10933-3) Proposes a method for estimating the probability that an individual has been successfully identified.
- [Clinical Text Anonymization, its Influence on Downstream NLP Tasks and the Risk of Re-Identification](https://aclanthology.org/2023.eacl-srw.11/) Re-ID patiens from their anonymized history.

## Attacks against Synthetic Data Generators
- [Synthetic Data -- Anonymisation Groundhog Day](https://arxiv.org/abs/2011.07018) Empirically show that synthetic data does not provide a better tradeoff between privacy and utility than traditional anonymisation techniques.
- [TAPAS: A toolbox for adversarial privacy auditing of synthetic data](https://arxiv.org/abs/2211.06550) Present a toolbox for performing attacks against synthetic data generators.
- [Achilles' Heels: Vulnerable Record Identification in Synthetic Data Publishing](https://arxiv.org/abs/2306.10308) Identify vulnerable records in the synthetic dataset.

## Data anonymization

- [Guaranteeing anonymity when sharing medical data, the Datafly System](https://pubmed.ncbi.nlm.nih.gov/9357587/#:~:text=We%20present%20a%20computer%20program,details%20found%20within%20the%20data.) - Foundational paper for k-anonymity
- [Automated anonymization of text documents](https://www.semanticscholar.org/paper/Automated-anonymization-of-text-documents-Mamede-Baptista/3c31662b2f81ad149c8e04025ad148b496d5118e) - Modular anonymization system for text documents.
- [Data Anonymization for Pervasive Health Care: Systematic Literature Mapping Study](https://pubmed.ncbi.nlm.nih.gov/34652278/) - Examines methods for text perturbation (like microaggregation or data swapping).
- [DataSifterText: Partially Synthetic Text Generation for Sensitive Clinical Notes](https://link.springer.com/article/10.1007/s10916-022-01880-6) - Use BERT to impute previously masked sensitive info.
- [Natural Text Anonymization Using Universal Transformer with a Self-attention](https://www.semanticscholar.org/paper/Natural-Text-Anonymization-Using-Universal-with-a-Romanov-Fedotova/acb2dfc52cb13762778f20a51c32071762afd867) - Anonymization system that uses a universal transformer model to generate anonymized text.
- [Deep Reinforcement Learning-based Text Anonymization against Private-Attribute Inference](https://aclanthology.org/D19-1240/) - Text anonymization system that manipulates the embeddings with an RL-based privacy-preserver.
- [Recovering from Privacy-Preserving Masking with Large Language Models](https://arxiv.org/abs/2309.08628) - Use an LLM to impute previously masked tokens.
- [Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection](https://arxiv.org/abs/2309.03057) - System using two local LLMs for anonymization and de-anonymization and in the middle use a black-box LLM.


## Data anonymization with Differential Privacy

- [Broadening the Scope of Differential Privacy Using Metrics](https://link.springer.com/chapter/10.1007/978-3-642-39077-7_5) - Foundational paper for Metric Differential Privacy
- [ADePT: Auto-encoder based Differentially Private Text Transformation](https://arxiv.org/abs/2102.01502) - Auto-encoder based DP algorithm to anonymize text while retaining utility. 
- [When differential privacy meets NLP: The devil is in the detail](https://arxiv.org/abs/2109.03175) - Formal analysis of ADePT, highlights some issues with privacy guarantees.
- [DP-VAE: Human-Readable Text Anonymization for Online Reviews with Differentially Private Variational Autoencoders](https://dl.acm.org/doi/abs/10.1145/3485447.3512232) - End-to-end DP-VAE for text anonymization.
- [DP-BART for Privatized Text Rewriting under Local Differential Privacy](https://arxiv.org/abs/2302.07636) - Text privatization model based on BART.
- [Sanitizing Sentence Embeddings (and Labels) for Local Differential Privacy](https://dl.acm.org/doi/abs/10.1145/3543507.3583512) - Sanitization system based on Purkayastha Mechanism.
- [Differential Privacy for Text Analytics via Natural Text Sanitization](https://arxiv.org/abs/2106.01221) - They propose SANTEXT to replace sensitive tokens.
- [A Customized Text Sanitization Mechanism with Differential Privacy](https://arxiv.org/abs/2207.01193) - They propose CUSTEXT to replace sensitive tokens.
- [InferDPT: Privacy-Preserving Inference for Black-box Large Language Model](https://cs.paperswithcode.com/paper/privinfer-privacy-preserving-inference-for) - They propose RANTEXT to replace sensitive tokens.
- [Leveraging Hierarchical Representations for Preserving Privacy and Utility in Text](https://arxiv.org/abs/1910.08917) - New noise distribution specifically devised for Metric DP.
- [A Differentially Private Text Perturbation Method Using a Regularized Mahalanobis Metric](https://arxiv.org/abs/2010.11947) - Regularized Mahalanobis Metric for text perturbation.
- [On a Utilitarian Approach to Privacy Preserving Text Generation](https://arxiv.org/abs/2104.11838) - Based on Vickrey auction, they balance the choice beetween first and second neighbours using a tuning parameter.
- [Guiding Text-to-Text Privatization by Syntax](https://arxiv.org/abs/2306.01471) - Includes grammatical categories into the privatization process to preserve syntax.
- [The Limits of Word Level Differential Privacy](https://arxiv.org/abs/2205.02130) - Paraphrasing model obtained by fine-tuning GPT2.

## Pre-training with Differential Privacy

- [Learning and Evaluating a Differentially Private Pre-trained Language Model](https://www.semanticscholar.org/paper/Learning-and-Evaluating-a-Differentially-Private-Hoory-Feder/c3b597011f64e7c5459bbe4502163e463fb13f5a) Fully private pre-training of BERT.
- [Learning and Evaluating a Differentially Private Pre-trained Language Model](https://arxiv.org/abs/2211.02956) Fully private pre-training of BERT for the legal domain.
- [Differentially Private Language Models Benefit from Public Pre-training](https://arxiv.org/abs/2009.05886) Comparison between fully private training and public pre-training with public fine-tuning for GPT-2.
- [Why Is Public Pretraining Necessary for Private Model Training?](https://arxiv.org/abs/2302.09483) Focused on the theoretical reasons why public pre-training is necessary for private learning.
- [https://arxiv.org/abs/2302.09483](https://arxiv.org/abs/2305.13865) Select pre-training data based on the fine-tuning data distribution, creating smaller pre-training datasets for smaller models.

## Fine-tuning with Differential Privacy

- [Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe](https://arxiv.org/abs/2210.14348) Private fine-tuning of GPT-2.
- [Making the Shoe Fit: Architectures, Initializations, and Tuning for Learning with Privacy](https://openreview.net/forum?id=rJg851rYwH) Propose different
architectures, initializations and hyperparameter tuning methods explicitly devised for private learning.
- [Simple Baselines Are Strong Performers for Differentially Private Natural Language Processing] (https://openreview.net/forum?id=oOiSJEr2-Tt) Ghost Clipping introduced to save memory and make private-learning almost on par with non-private-learning from a memory usage point of view.
- [Differentially Private Optimization on Large Model at Small Cost](https://arxiv.org/abs/2210.00038) New Book-Keeping technique that requires a single backpropagation pass.
- [Differentially Private Language Models for Secure Data Sharing](https://arxiv.org/abs/2210.13918) DP-tuning of GPT-2 to generate a synthetic and private version of the tuning dataset.
- [EW-Tune: A Framework for Privately Fine-Tuning Large Language Models with Differential Privacy](https://arxiv.org/abs/2210.15042) Decrease the induced noise significantly by using Edgeworth accountant and (realistically) assuming that tuning epochs are not many.
  
## Parameter-Efficient Fine-Tuning with Differential Privacy

- [Differentially Private Fine-tuning of Language Models](https://arxiv.org/abs/2110.06500) DP-PEFT of both RoBERTa and GPT-2 with several techniques.
- [Large Language Models Can Be Strong Differentially Private Learners](https://arxiv.org/abs/2110.05679) DP-PEFT of both RoBERTa and GPT-2 with several techniques.
- [Privacy-Preserving Prompt Tuning for Large Language Model Services](https://arxiv.org/abs/2305.06212) DP-prompt-tuning framework (RAPT).

## Reinforcement Learning with Differential Privacy

- [Privately Aligning Language Models with Reinforcement Learning](https://arxiv.org/abs/2310.16960) DP-RL of GPT-2

## Inference with Differential Privacy

- [Privacy-Preserving In-Context Learning for Large Language Models](https://arxiv.org/abs/2305.01639) DP-ICL based on partitioning the dataset, use it to get ICL examples, then aggregate the partitioned answers with noise to get a final answer.
- [Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation](https://arxiv.org/abs/2309.11765) DP-ICL based on partitioning the dataset, use it to get ICL examples, then aggregate the partitioned answers with noise to get a synthetic ICL example.
- [Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models](https://arxiv.org/abs/2305.15594) Prompt-PATE is used here, where an ensemble of teachers get private samples to give private answers that are later noisily aggregated into a synthetic (private) example.
- [DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer](https://arxiv.org/abs/2312.03724) DP-Offsite Prompt Tuning uses an ensemble of local models to get private predictions and then aggregate them with noise.
- [Split-and-Denoise: Protect large language model inference with local differential privacy](https://openreview.net/forum?id=vxmvbzw76R&referrer=%5Bthe%20profile%20of%20Peihua%20Mai%5D(%2Fprofile%3Fid%3D~Peihua_Mai1)) Local encoder and decoder to add noise to the input and remove noise from the output of an offsite LLM.
- [InferDPT: Privacy-Preserving Inference for Black-box Large Language Model](https://cs.paperswithcode.com/paper/privinfer-privacy-preserving-inference-for) InferDPT has local anonymizer and de-anonymizer to anonymize the input and de-anonymize the output of an offsite LLM.

## Federated Learning with Differential Privacy

- [Training Production Language Models without Memorizing User Data](https://arxiv.org/abs/2009.10031) Next word prediction model trained in a federated fashion with DP-FedAVG
- [Can Public Large Language Models Help Private Cross-device Federated Learning?](https://arxiv.org/abs/2305.12132) Introduce DP-FTRL and use LLMs to improve privacy/utility tradeoff of the local LM in DP-FL.
- [Federated Learning of Gboard Language Models with Differential Privacy](https://arxiv.org/abs/2305.18465) Present and analyze twenty Gboard LMs trained for Next Word Prediction with DP-FTRL.
- [Benchmarking Differential Privacy and Federated Learning for BERT Models](https://arxiv.org/abs/2106.13973) DP-FL training of BERT, RoBERTa, DistillBERT and ALBERT.

## Machine Unlearning

- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) ROME method allows to trace factual predictions back to single neurons and manipulate them.
- [Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks](https://arxiv.org/abs/2309.17410) Shows issues of ROME and improves upon it.
- [DEPN: Detecting and Editing Privacy Neurons in Pretrained Language Models](https://arxiv.org/abs/2310.20138) Detect and manipulate neurons connected with private information.
- [Machine Unlearning](https://arxiv.org/abs/1912.03817) SISA method divides training data and the model training so that unlearning involves repeating just a part of the training process.
- [Knowledge Unlearning for Mitigating Privacy Risks in Language Models](https://arxiv.org/abs/2210.01504) Negates the loss function used in training, with the objective to maximize the loss on the target sequences.
- [Who's Harry Potter? Approximate Unlearning in LLMs](https://arxiv.org/abs/2310.02238) Use a model fine-tuned on the data to forget in order to compare with the original model the likelihood growth and identify the sensitive data.
- [Unlearn What You Want to Forget: Efficient Unlearning for LLMs](https://arxiv.org/abs/2310.20150) Build unlearning layers and train them with a selective student-teacher objective based on KL-divergence in order for the student model to maximize the
divergence from the teacher model (on target data).
- [Preserving Privacy Through Dememorization: An Unlearning Technique For Mitigating Memorization Risks In Language Models](https://aclanthology.org/2023.emnlp-main.265/) DeMemorization through Reinforcement Learning.
- [Knowledge Sanitization of Large Language Models](https://arxiv.org/abs/2309.11852) Sanitization approach to limit hallucinations deriving from unlearning.
- [In-Context Unlearning: Language Models as Few Shot Unlearners](https://arxiv.org/abs/2310.07579) Machine Unlearning enforced through ICL.

## Tools and Frameworks

- [TensorFlow Privacy](https://github.com/tensorflow/privacy) Python library with optimizers for training ML models with DP.
- [PyVacy](https://github.com/ChrisWaites/pyvacy) Pytorch translation of TensorFlow Privacy.
- [OpenDP project](https://projects.iq.harvard.edu/files/opendp/files/opendp_white_paper_11may2020.pdf) Collection of algorithms for generating DP statistics.
- [DiffPrivLib](https://api.semanticscholar.org/CorpusID:195798910) Provides a wide range of DP tools for ML and data analysis.
- [Google DP](https://github.com/google/differential-privacy) Provides a broad set of DP tools.
- [Microsoft DP](https://arxiv.org/abs/2309.11765) Inference-DP framework.
- [EKTELO](https://dl.acm.org/doi/10.1145/3183713.3196921) Flexible and extensible framework for DP data analysis.
- [PyTorch Opacus](https://arxiv.org/abs/2109.12298) Enables training PyTorch models with DP.
- [private-transformers](https://github.com/lxuechen/private-transformers) Provides a privacy engine built off Opacus rewritten specifically to facilitate integration with the transformers library.
- [dp-transformers](https://github.com/microsoft/dp-transformers) Toolkit that provides a simplified integration of transformers training with DP.
- [Chorus](https://api.semanticscholar.org/CorpusID:226266222) DP statistical queries through a cooperative query processing system.
- [autodp](https://github.com/yuxiangw/autodp) Automates the process of calculating the privacy guarantees for complex algorithms and supports several standard DP mechanisms.
