# TinyZero

> **⚠️ Deprecation Notice:** This repo is no longer actively maintained. For running RL experiments, please directly use the latest [veRL](https://github.com/volcengine/verl) library.
> For the archived original documentation, see [OLD_README.md](./OLD_README.md).

![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own.

You can experience the Aha moment yourself for < $30.

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

> 📢: We release [Adaptive Parallel Reasoning](https://github.com/Parallel-Reasoning/APR), where we explore a new dimension in scaling reasoning models.

## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Countdown task

**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

### Run Training
```
conda activate zero
```

For the following code, if you see out-of-VRAM, try adding `critic.model.enable_gradient_checkpointing=True` to the script, and check out the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643).

**Single GPU**


Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation
We experiment with Qwen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Acknowledgements
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
