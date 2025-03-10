# Li_MultiView_reproduction_tensorflow
This repository using tensorflow to reproduce an multi-view DL-based approach to
segment the claustrum in T1-weighted MRI scans. 

Reference of the original GitHub: https://github.com/hongweilibran/claustrum_multi_view

Reference of the original paper: https://arxiv.org/abs/1911.07515


## Overview
```bibtex
@article{albishri2019automated,
  title={Automated human claustrum segmentation using deep learning technologies},
  author={Albishri, Ahmed Awad and Shah, Syed Jawad Hussain and Schmiedler, Anthony and Kang, Seung Suk and Lee, Yugyung},
  journal={arXiv preprint arXiv:1911.07515},
  year={2019}
}
```


## Environments and Requirements

This test implementation is designed to run on **CPU**.

**Python**: Version 3.12

To set up the environment:
```bash
git clone https://github.com/ShutingXie/Li_MultiView_reproduction_tensorflow.git
```

Create a virtual environment (You can use other way to creat a virtual environment):
```bash
python -m venv myenv
```

Activate the virtual environment:
```bash
source myenv/bin/activate
```

Install all dependent packages
```bash
pip install -r requirements.txt
```


## Dataset
Put your MRI data and labels in the **data_org/** and **labels_org** folders respectively


## Test
```bash
python test.py
```

## Evaluation the prediction results
```bash
python compute_agreement.py
```





   
