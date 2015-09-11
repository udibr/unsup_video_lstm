Code for [Unsupervised Learning of Video Representations using LSTMs](http://arxiv.org/pdf/1502.04681.pdf)
taken from http://www.cs.toronto.edu/~nitish/unsupervised_video/

INSTALL
=======

```bash
export CUDA_ROOT=...
make
mkdir mnist
cd mnist
wget www.cs.toronto.edu/~nitish/data/mnist.h5
mkdir models
cd ..
```

RUN
===
```
python lstm_composite.py lstm_combo_1layer_mnist.pbtxt bouncing_mnist.pbtxt
```
