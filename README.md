# Enhancing-RAW-to-sRGB-with-Decoupled-Style-Structure-in-Fourier-Domain
This repository contains the implementation for the method described in the paper titled "Enhancing RAW-to-sRGB with Decoupled Style Structure in Fourier Domain." The paper can be found at: https://arxiv.org/abs/2401.02161.

## overview
This method introduces a novel approach for enhancing RAW-to-sRGB conversion by leveraging decoupled style structure in the Fourier domain. The proposed technique aims to improve the visual quality and fidelity of the generated sRGB images from RAW sensor data.

## usage
The checkpoint for this method is not publicly available. However, users can reproduce the results by running the provided training script:
```
train.py -opt options/train/FourierISP/train_FourierISP.yml
```
## Data
For unaligned datasets, we recommend using the ZRR dataset. For aligned datasets, the aligning process can be implemented using either alignformer or liteisp.
Please note that the alignment of datasets is crucial for obtaining best visual results in the training process.



