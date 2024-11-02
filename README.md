# InterpreFFA
We developed a diagnosis-supervised contrastive learning framework named InterpreFFA to emulate the decision-making process of ophthalmologists on generating FFA reports in real-world clinical practice.

This is the pytorch implementation for our paper.

## Requirements

- `torch>=1.6.0`
- `torchvision>=0.8.0`

## Datasets
We use three datasets in the paper: the Second Affiliated Hospital Zhejiang University School of Medicine (ZJU2, internal dataset), Taizhou First People’s Hospital and the Second Affiliated Hospital of Xi’an Jiaotong University (TZ and XJU2, external datasets). 

Data will be made available for research purposes upon request. Data requests are to be directed to KJ.

## Run on ZJU2 dataset

- Run `bash train_zju2.sh` to train our model on the ZJU2 dataset.
  
- Run `bash test_zju2.sh` to test our model on the ZJU2 dataset.

## Acknowledgments

This project is built on top of [R2Gen](https://github.com/cuhksz-nlp/R2Gen). Thank the authors for their contributions to the community!
