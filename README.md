# H3M-SSMoEs: Hypergraph-based Multimodal Learning with LLM Reasoning and Style-Structured Mixture of Experts

The repo is the official implementation for the paper: [H3M-SSMoEs: Hypergraph-based Multimodal Learning with LLM Reasoning and Style-Structured Mixture of Experts](https://arxiv.org/abs/2510.25091).
.

## Introduction 📖
H3M-SSMoEs is a hypergraph-based multimodal framework combining LLM reasoning and style-structured experts for stock prediction.

### Overall Architecture
![Model Framework](https://github.com/PeilinTime/H3M-SSMoEs/blob/main/figure/overview%20of%20H3M-SSMoEs.png)

---




## Datasets & Model Weights 📦

All datasets, including quantitative features, news embeddings, and timestamp embeddings, are available via Google Drive. 👉 [Datasets Download Link](https://drive.google.com/drive/folders/1kJobHYib_WSwQHHU958sh0gKRgyA-Lu7?usp=sharing)

The corresponding model weights are hosted on MEGA. 👉 [Model Weights Download Link](https://mega.nz/folder/HMdSySAQ#jtR8Y5BdtmtCr3XhrxYnQQ)

Included datasets and weights:

* **DJIA**
* **NASDAQ100**
* **S&P100**

---

## How to Run Our Model 🚀

### 1. Download this repository

Download or clone this code repository to your local machine.

### 2. For Training and Prediction

Download one of the datasets from the Google Drive link above:

* DJIA: `djia_alpha158_alpha360.pt` & `djia_news_embeddings.pt` & `timestamps_embedding.pt`
* NASDAQ100: `nas100_alpha158_alpha360.pt` & `nas100_news_embeddings.pt` & `timestamps_embedding.pt`
* S&P100: `sp100_alpha158_alpha360.pt` & `sp100_news_embeddings.pt` & `timestamps_embedding.pt`

Place the downloaded files in the same directory as the codebase.
Run the following command to train the model and make predictions (including training, validation, and test sets):

```bash
python run.py
```

### 3. For Backtesting

Download the datasets **and** the corresponding model weights from the two links above:

* DJIA: `djia_alpha158_alpha360.pt` & `djia_news_embeddings.pt` & `timestamps_embedding.pt` **and** `DJIA_weight.pth`
* NASDAQ100: `nas100_alpha158_alpha360.pt` & `nas100_news_embeddings.pt` & `timestamps_embedding.pt` **and** `NASDAQ100_weight.pth`
* S&P100: `sp100_alpha158_alpha360.pt` & `sp100_news_embeddings.pt` & `timestamps_embedding.pt` **and** `SP100_weight.pth`

Place the downloaded files in the same directory as the codebase.
Run the following command to perform backtesting and results:

```bash
python backtesting.py
```

---

## Backtesting Results 📈

Below are the backtesting performance charts of our model on all datasets:

![Backtesting_result_DJIA](https://github.com/PeilinTime/H3M-SSMoEs/blob/main/figure/Backtesting_result_DJIA.png)
![Backtesting_result_NASDAQ100](https://github.com/PeilinTime/H3M-SSMoEs/blob/main/figure/Backtesting_result_NASDAQ100.png)
![Backtesting_result_S&P100](https://github.com/PeilinTime/H3M-SSMoEs/blob/main/figure/Backtesting_result_SP100.png)

## Citation
We would appreciate it if you could cite the following paper if you found the repository useful for your work:

```bash
@misc{tan2025h3mssmoeshypergraphbasedmultimodallearning,
      title={H3M-SSMoEs: Hypergraph-based Multimodal Learning with LLM Reasoning and Style-Structured Mixture of Experts}, 
      author={Peilin Tan and Liang Xie and Churan Zhi and Dian Tu and Chuanqi Shi},
      year={2025},
      eprint={2510.25091},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.25091}, 
}
```
