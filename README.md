# Zero-shot-KPE-with-LLMs
This is code of our paper "[Empirical Study of Zero-shot Keyphrase Extraction with Large Language](https://aclanthology.org/2025.coling-main.248)".

## Requirements
- torch=2.2.2
- transformers==4.40.0
- nltk 3.7
- StanfordCoreNLP 3.9.1.1
- [stanford-corenlp-full-2018-02-27](https://drive.google.com/file/d/1K4Ll54ypTf_tF83Mkkar2QKOcZ4Uskl5/view?usp=sharing)


## Run
```python
# Vanilla
python llama3_vanilla.py \
  --auth_token "your huggingface auth_token for Llama3"


# Candidate
python llama3_candidate.py \
  --core_nlp_path "your StanfordCoreNLP path"\
  --auth_token "your huggingface auth_token for Llama3"


# Hybrid
python llama3_hybrid.py \
  --core_nlp_path "your StanfordCoreNLP path"\
  --auth_token "your huggingface auth_token for Llama3"
```



## Evaluation
```python
python evaluation.py --path "directory path of prediction files"
```



## Citation
If you use this code, please cite our paper: 
```
@inproceedings{kang-shin-2025-empirical,
    title = "Empirical Study of Zero-shot Keyphrase Extraction with Large Language Models",
    author = "Kang, Byungha  and
      Shin, Youhyun",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.248/",
    pages = "3670--3686",
    abstract = "This study investigates the effectiveness of Large Language Models (LLMs) for zero-shot keyphrase extraction (KE). We propose and evaluate four prompting strategies: vanilla, role prompting, candidate-based prompting, and hybrid prompting. Experiments conducted on six widely-used KE benchmark datasets demonstrate that Llama3-8B-Instruct with vanilla prompting outperforms state-of-the-art unsupervised methods, PromptRank, by an average of 9.43{\%}, 7.68{\%}, and 4.82{\%} in F1@5, F1@10, and F1@15, respectively. Hybrid prompting, which combines the strengths of vanilla and candidate-based prompting, further enhances overall performance. Moreover role prompting, which assigns a task-related role to LLMs, consistently improves performance across various prompting strategies. We also explore the impact of model size and different LLM series: GPT-4o, Gemma2, and Qwen2. Results show that Llama3 and Gemma2 demonstrate the strongest zero-shot KE performance, with hybrid prompting consistently enhancing results across most LLMs. We hope this study provides insights to researchers exploring LLMs in KE tasks, as well as practical guidance for model selection in real-world applications. Our code is available at https://github.com/kangnlp/Zero-shot-KPE-with-LLMs."
}
```
