# Multi-Vector Search: Evaluating Chamfer versus other rankers

I want to be able to keep track of where I got the datasets from and how I ran everything in this codebase.

## Datasets
Create a directory called `datasets` at the top level of the repository. Run these inside that directory.

- arguana
    - Corpus + Queries: `git clone https://huggingface.co/datasets/BeIR/arguana`
    - Ground Truth: `git clone https://huggingface.co/datasets/BeIR/arguana-qrels`
- quora
    - Corpus + Queries: `git clone https://huggingface.co/datasets/BeIR/quora`
    - Ground Truth: `git clone https://huggingface.co/datasets/BeIR/quora-qrels`
- scidocs
    - Corpus + Queries: `git clone https://huggingface.co/datasets/BeIR/scidocs`
    - Ground Truth: `git clone https://huggingface.co/datasets/BeIR/scidocs-qrels`
- OBLIQ Datasets
    - Run `python ../src/convert_obliq.py` to get the datasets in the BEIR format
## Multi-Vector Encodings
None of these datasets come with pre-computed multi-vector encodings. I produce them myself using BERT and ColBERT. 

Punch in the dataset path(s) and the desired method and the following command should build out the vectors for you. 
```
python src/encode.py --method [bert | colbert] --datasets [DATASETS]
```

