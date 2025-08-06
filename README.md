# SpatialFinder



### segmentation model Description

This repository used a **complete copy of the official nnU-Net v2 code** (original source: https://github.com/MIC-DKFZ/nnUNet). The file structure is explained below:

| File/Directory   | Purpose                                                      |
| :--------------- | :----------------------------------------------------------- |
| `nnunetv2/`      | **Official core code**: Model architecture and training/inference pipeline (Copyright © MIC-DKFZ) |
| `.gitignore`     | Repository configuration: Excludes temporary files (`__pycache__`) and sensitive data |
| `LICENSE`        | **Original license**: Apache 2.0 (as per official repository) |
| `pyproject.toml` | Dependency configuration: Declares required Python environment |
| `setup.py`       | Installation script: Supports deployment via `pip install .` |
| `README.md`      | Project documentation (this file)                            |
| `run_nnunet.sh`  | **Reference script**: Example usage of official commands (non-official component) |
| `tst.sh`         | Test script: Environment verification (non-official component) |

#### Important Disclaimers

1. **Code Ownership**
   All code in the `nnunetv2/` directory is the original work of the [German Cancer Research Center (MIC-DKFZ)](https://www.dkfz.de/en/mic/index.php). This repository only hosts an unmodified copy.

2. **Modification Record**
   No changes were made to the core code. Only peripheral configurations (e.g., sample scripts, `.gitignore`) were adjusted.

3. **Official Documentation**
   Full documentation: [nnU-Net Official Docs](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation.md)

    

   

###  Classification model Description

All code in the `UNI` directory is the original work of the Nat Med (2024). https://doi.org/10.1038/s41591-024-02857-3. This repository only hosts an unmodified copy.https://github.com/mahmoodlab/UNI

### Requirements

PyTorch: 2.1.1

Python 3.9

CUDA Toolkit: 11.8

cuDNN: 8.6.0

transformers: 4.30.0

scikit-learn: 1.0.0

torchvision: 0.16.1

Pillow: 8.0.0

opencv-python: 4.5.0

torch==2.1.1+cu118

torchvision==0.16.1+cu118

numpy>=1.21.0

pandas>=1.3.0

scikit-learn>=1.0.0

tqdm>=4.64.0

xgboost==1.7.0 

matplotlib==3.5.3 

seaborn==0.12.2

```

```

