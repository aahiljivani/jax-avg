# jax-avg

A JAX based implementation of the **Action Value Gradient. (AVG)** This is an incremental deep policy gradient method that learns from each sample without batch updates, target networks, or replay buffers.

The original PyTorch implementation is here: [https://github.com/gauthamvasan/avg](https://github.com/gauthamvasan/avg).

This is a **streamRL** (streaming / incremental RL) algorithm. For method details, see the paper: [Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers](https://arxiv.org/pdf/2411.15370).

## Setup

```bash
conda create -n jax-avg python=3.10 pip -y
conda activate jax-avg
pip install -r requirements.txt
```



## Train

```bash
python train.py
```

By default this trains on Brax `walker2d` for 1M+ timesteps. Common flags:

```bash
python train.py --env walker2d --N 10001000 --seed 42 --num_envs 1000
```



## Weights & Biases

Training logs episode return and losses to [Weights & Biases](https://wandb.ai) by default. Disable it with `--no-wandb`.
