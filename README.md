# 3dDNAscoreA



## Intro

3dDNAscoreA is an all-atom energy function 3dDNAscoreA for evaluation of DNA 3D structures based on a deep learning model of ARES for RNA 3D structure evaluation but uses a new training strategy that uses the DNA training sets within a RMSD thresholds to train the model,  which is an extension of our RNA 3D structure prediction method 3dRNA (http://biophy.hust.edu.cn/new/3dRNA/create).



## Run 3dDNAscoreA for prediction

For a DNA decoys, the general prediction format is as follow:

`python -m ares.predict pdb_dir model_dir output.csv -f [pdb|silent|lmdb] [--nolabels] [--label_dir=score_dir]`

For example, to predict the DNA 1EZN containing 500 decoys:

`python -m ares.predict dataset/TestII/decoys/1EZN Model/epoch50/Model_6/model6.ckpt 1EN1.csv -f pdb --nolabels --gpus=1 --num_workers=8`



## Requirements and Installation

The code base is implemented in Python 3.8. The package versions used for development are mainly as follows:

```
pytorch-lightning       1.0.0 
python-dotenv           0.19.0
wandb                   0.15.2
atom3d                  v0.2.4
e3nn                    0.1.0
torch                   1.5.0+cu101
torch-cluster           1.5.7
torch-geometric         1.7.0
torch-scatter           2.0.4
torch-sparse            0.6.7
torch-spline-conv       1.2.0
torchvision             0.6.0+cu101
```

Configure the environment according to the yml file:

`conda env create -f env_name.yml`

