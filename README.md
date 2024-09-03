# Starter code on discrete diffusion
This is a starter repo that contains code to train, sample and evaluate MDLM/SEDD discrete diffusion models.

## Install dependencies
```bash
mamba create -n fast_discrete_diff python=3.10 -y
mamba activate fast_discrete_diff
pip install -r requirements.txt
pip install flash_attn
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu

pip install -e .
```

## Train an MDLM model
```bash
python src/starter_discrete_diff/main.py \
    mode=train \
    parameterization=mdlm \
    model=tiny \
    data=wikitext2 \
    time_conditioning=false \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    trainer.max_steps=10000 \
    eval_every=1000 \
    hydra.run.dir="./outputs/lm1b/mdlm_small" \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    loader.num_workers=8 \
    compile=False \
    wandb.project="debug" \
    wandb.notes="train small mdlm" \
    trainer.precision="bf16-mixed" \
```


## Sample from the trained model
```bash
python src/starter_discrete_diff/main.py \
    mode=sample \
    parameterization=mdlm \
    model=tiny \
    data=wikitext2 \
    time_conditioning=false \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    trainer.max_steps=10000 \
    eval_every=1000 \
    hydra.run.dir="./outputs/lm1b/mdlm_small" \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    loader.num_workers=8 \
    compile=False \
    wandb.project="debug" \
    wandb.notes="train small mdlm" \
    trainer.precision="bf16-mixed" \
    \
    parameterization.sampling.uncond.run=True \
    parameterization.sampling.uncond.num_samples=8 \
    parameterization.sampling.uncond.batch_size=8 \
    parameterization.sampling.uncond.num_steps=64 \
    \
    parameterization.sampling.cond_prefix.run=True \
    parameterization.sampling.cond_prefix.num_samples=8 \
    parameterization.sampling.cond_prefix.batch_size=8 \
    parameterization.sampling.cond_prefix.num_steps=64 \


```

## Sample-based evaluation
```bash
python src/starter_discrete_diff/main.py \
    mode=eval \
    parameterization=mdlm \
    model=tiny \
    data=wikitext2 \
    time_conditioning=false \
    loader.global_batch_size=8 \
    loader.batch_size=8 \
    trainer.max_steps=10000 \
    eval_every=1000 \
    hydra.run.dir="./outputs/lm1b/mdlm_small" \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    loader.num_workers=8 \
    compile=False \
    wandb.project="debug" \
    wandb.notes="train small mdlm" \
    trainer.precision="bf16-mixed" \
    \
    eval.ppl_with_ar.run=True \
    eval.ppl_with_ar.batch_size=8 \
```

## Resources
- [MDLM](https://s-sahoo.com/mdlm/)
- [SEDD](https://aaronlou.com/blog/2024/discrete-diffusion/)