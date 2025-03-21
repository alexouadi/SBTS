# 📌 Official Implementation of "Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation"

This repository contains the official implementation of the paper ["Robust Time Series Generation via Schrödinger Bridge: A Comprehensive Evaluation"](https://arxiv.org/abs/2503.02943).

Authors: Alexandre Alouadi, Baptiste Barreau, Laurent Carlier, Huyên Pham

Contact: alexandre.alouadi@bnpparibas.com; huyen.pham@polytechnique.edu

If you notice any errors or have suggestions for improvement, please feel free to reach out to us.

## Abstract
We investigate the generative capabilities of the Schrödinger Bridge (SB) approach for time series. The SB framework formulates time series synthesis as an entropic optimal interpolation transport problem between a reference probability measure on path space and a target joint distribution. This results in a stochastic differential equation over a finite horizon that accurately captures the temporal dynamics of the target time series. While the SB approach has been largely explored in fields like image generation, there is a scarcity of studies for its application to time series. In this work, we bridge this gap by conducting a comprehensive evaluation of the SB method's robustness and generative performance. We benchmark it against state-of-the-art (SOTA) time series generation methods across diverse datasets, assessing its strengths, limitations, and capacity to model complex temporal dependencies. Our results offer valuable insights into the SB framework's potential as a versatile and robust tool for time series generation.

## 📂 Project Structure
```
/SBTS
│── data                    # Real-world datasets used in experiments
│── metrics                 # Evaluation metrics as described in the paper
│── models                  # Core modules for Schrödinger Bridge-based time series generation
│── utils                   # Utility functions
│── experiments_demo.ipynb  # Example usage and evaluation
```

## Setup
To clone the repository, run:
```bash
git clone https://github.com/alexouadi/SBTS.git
cd SBTS
```

To install the necessary dependencies for this project, make sure you use python 3.11 and run:
```bash
pip install -r requirements.txt
```

## Usage
You can use the implemented functions directly by importing them into your script, run the Jupyter notebook to see practical examples:
```bash
jupyter notebook demo_notebook.ipynb
```

## Bibtex

```bibtex
@misc{alouadi2025robusttimeseriesgeneration,
      title={Robust time series generation via Schr\"odinger Bridge: a comprehensive evaluation}, 
      author={Alexandre Alouadi and Baptiste Barreau and Laurent Carlier and Huyên Pham},
      year={2025},
      eprint={2503.02943},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.02943}, 
}


