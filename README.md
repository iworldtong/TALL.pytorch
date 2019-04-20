# TALL.pytorch
PyTorch implementation of ["TALL: Temporal Activity Localization via Language Query. Gao et al. ICCV2017."](<https://arxiv.org/abs/1705.02101>)

This implementation highly based on official code [jiyanggao/TALL](<https://github.com/jiyanggao/TALL>).(Tensorflow)

### Require

- Python 3.6
- PyTorch 1.0
- CPU | GPU supported.

### Visual Features on TACoS

Download the C3D features for [training set](https://drive.google.com/file/d/1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu/view?usp=sharing) and [test set](https://drive.google.com/file/d/1zC-UrspRf42Qiu5prQw4fQrbgLQfJN-P/view?usp=sharing) of TACoS dataset. Modify the path to feature folders in `config.py`.

### Sentence Embeddings on TACoS

Download the Skip-thought sentence embeddings and sample files from [here](https://drive.google.com/file/d/1HF-hNFPvLrHwI5O7YvYKZWTeTxC5Mg1K/view?usp=sharing) of TACoS Dataset, and put them under exp_data folder.

### Reproduce the results on TACoS

`python main.py`

