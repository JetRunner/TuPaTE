# TuPaTE
Code for EMNLP 2022 paper ["Efficiently Tuned Parameters are Task Embeddings"](https://arxiv.org/abs/2210.11705)
<img width="752" alt="Workflow of TuPaTE." src="https://user-images.githubusercontent.com/22514219/205440825-1b8b074f-5acc-44be-994f-38a0a1f21098.png">

### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment by:

```shell
conda create -n tupate python=3.8.5
conda activate tupate
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

### Training
Run training scripts in [run_script](run_script) (e.g., RoBERTa for RTE):

```shell
bash run_script/run_rte_bert.sh
```

### Extract Task Embedding

Functions for extracting task embeddings for different parameter efficient tuning methods are provided in
```shell
extract_task_emb.py
```

### Pre-computed Task Embedding

We also release the embeddings for each task [here](https://github.com/JetRunner/TuPaTE/tree/main/task_embeddings).
