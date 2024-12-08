# virt_final

## PyTorch
```bash
conda create -n your_env pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate your_env
python pytorch.py
```

## Tensorflow
```bash
conda create -n your_env tensorflow-gpu nccl -c conda-forge
conda activate your_env
XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX python tf.py --gpus 4
```


