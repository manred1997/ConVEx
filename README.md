# ConVEx: Data-Efficient and Few-Shot Slot Labeling


Details of our ConVEx model architecture, dataset construction and experimental results can be found in our [following paper](https://arxiv.org/pdf/2010.11791v2.pdf):

## Dataset


## Model installation, training and evaluation

### Installation
- Python version >= 3.6
- PyTorch version >= 1.4.0

```
    git clone https://github.com/manred1997/ConVEx.git
    cd ConVEx
    pip3 install -r requirements.txt
```

### Pretraining and Evaluate Model
Run the following two bash files to reproduce results presented in our paper:
```
    ./main.sh
```

 - Here, in these bash files, we include running scripts to train both our ConVEx and the ConVEx+CRF.  

Please prepare your data with the same format as in the ```data``` directory.

