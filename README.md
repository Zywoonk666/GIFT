# GIFT

This is the codebase for **GIFT**, which is built and improved upon the GIMLET framework for molecule learning. GIFT is a unified graph-text model tailored and evaluated focusing deeply on **Few-Shot Learning** for downstream tasks.

## Installation

To run GIFT, please install the required dependencies using the script provided.
Please note that the environment initialization script is for CUDA 11.1. If you are using a different version of CUDA, please adjust the package version in the script as necessary.

### Environment
```bash
conda create -n env_gimlet python=3.7
source activate env_gimlet

pip3 install torch==1.9+cu111 torchaudio  -f https://download.pytorch.org/whl/cu111/torch_stable.html

wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl

pip install torch_geometric==1.7.2

git clone -b v4.28.1 https://github.com/huggingface/transformers
cd transformers
pip install --editable ./
python setup.py build_ext --inplace
cd ..

pip install datasets evaluate ogb spacy tqdm sklearn SentencePiece lmdb tensorboardX==2.4.1 rdkit-pypi==2021.9.3 wandb networkx setuptools==59.5.0 multiprocess Cython plotly commentjson pandas boto3 chainer-chemistry subword-nmt pytorch_lightning==1.0 -U kaleido openai
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
```

## Dataset Download

The MoleculeNet datasets (including pcba, bace, hiv, muv, tox21, toxcast, bbbp, esol, lipo, and freesolv) can be downloaded manually or automatically upon the first run. 
To download manually:
```bash
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset property_data
```

## Run Few-Shot Learning Experiments

We provide an integrated script to run the few-shot learning pipeline, which evaluates the model natively on downstream tasks.

To execute the reproduction of the few-shot learning downstream task, run:
```bash
bash run_experiments.sh
```

The script automates few-shot testing (e.g. on `bace`) leveraging the GIFT custom architecture. All evaluation results will be cached as CSV files in the `./cache/` directory.

## Citation (Base Model)

Since GIFT is formulated upon the GIMLET model, please consider citing their original paper if you find the base framework helpful:
```bibtex
@article{zhao2024gimlet,
  title={Gimlet: A unified graph-text model for instruction-based molecule zero-shot learning},
  author={Zhao, Haiteng and Liu, Shengchao and Chang, Ma and Xu, Hannan and Fu, Jie and Deng, Zhihong and Kong, Lingpeng and Liu, Qi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
