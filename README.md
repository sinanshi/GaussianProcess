# GaussianProcess

[![Build Status](https://travis-ci.org/sinanshi/GaussianProcess.svg?branch=master)](https://travis-ci.org/sinanshi/GaussianProcess)

GPU Gaussian process emulator



# Installation
* Requirement: 
  * Python3
  * scipy 
  * numpy
  * CUDA
  * [MAGMA](http://icl.cs.utk.edu/magma/)

* Steps:

MAGMA is not currently available on JADE. Therefore one must specify the 
MAGMA library path before compiling. 

```bash
export MAGMAHOME=/jmain01/home/JAD013/sxg01/sxs32-sxg01/magma/
```

Install the python package

```bash
cd GaussianProcess
python setup.py install
```

# Examples 
See example.py 
