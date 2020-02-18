# 辅助软件


## google trax
- [google trax](https://github.com/google/trax)

pip 安装

```
pip install trax==1.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

#重新安装jaxlib的GPU版本
# install jaxlib
PYTHON_VERSION=cp37  # alternatives: cp35, cp36, cp37, cp38
CUDA_VERSION=cuda92  # alternatives: cuda92, cuda100, cuda101, cuda102
PLATFORM=linux_x86_64  # alternatives: linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.39-$PYTHON_VERSION-none-$PLATFORM.whl

pip install --upgrade jax  # install jax

```


## reformers_python 

```
pip install reformer_pytorch
```