# Learning Compound Tasks via Imitation and Self-supervised Learning

PyTorch code for the submission:
>Learning Compound Tasks without Task-specific Knowledge via Imitation and Self-supervised Learning, ICML 2020

Since the code used for our experiments is not publicly available due to my company's internal restrictions on code release,
we reimplemented the code for our paper based on the open source (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).
We confirmed that this code can produce very similar results to the experimental results in our paper.
We also provide implementations of the newly introduced tasks, which we call MountainToyCar-v1 and MountainToyCarContinuous-v1. 
Further details are described below.

## Requirements 

- Python 3
- PyTorch
- OpenAI baselines

Other requirements can be installed using pip by running:
```bash
pip install -r requirements.txt
``` 

## Setup for Experiments
In order to install compound tasks introduced in this paper, please run following commands:
```bash
cd custom-tasks
pip install -e .
```

## Training
#### MountainToyCar-v1
```bash
python main.py --env-name "MountainToyCar-v1" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 \
--lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 \
--num-env-steps 1026048 --use-linear-lr-decay --use-proper-time-limits \
--gail --extract-obs --gail-algo standard --expert-algo ppo \
--use-latent --latent-dim 1 --hierarchical-policy --task-transition --posterior \
--save-date 200219 --eval-interval 1
```

#### MountainToyCarContinuous-v1
```bash
python main.py --env-name "MountainToyCarContinuous-v1" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 \
--lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 \
--num-env-steps 1026048 --use-linear-lr-decay --use-proper-time-limits \
--gail --extract-obs --gail-algo standard --expert-algo ppo \
--use-latent --latent-dim 1 --hierarchical-policy --task-transition --posterior \
--save-date 200219 --eval-interval 1
```

#### FetchPickAndPlace-v1
```bash
python main.py --env-name "FetchPickAndPlace-v1" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 \
--lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 \
--num-env-steps 1230848 --use-linear-lr-decay --use-proper-time-limits \
--gail --gail-algo standard --expert-algo her --load-model --load-date 200219 --pretrain-algo cvae \
--task-transition --use-latent --latent-dim 4 --hierarchical-policy --posterior \
--reset-posterior --reset-transition --save-date 200219 --eval-interval 1
```

## Testing
#### MountainToyCar-v1
```bash
python enjoy.py --env-name MountainToyCar-v1 --algo ppo --gail-algo standard --pretrain-algo none --test-model trained \
--episode 100 --use-latent --latent-dim 1 --task-transition --save-date 200219 --load-date 200219
```
#### MountainToyCarContinuous-v1
```bash
python enjoy.py --env-name MountainToyCarContinuous-v1 --algo ppo --gail-algo standard --pretrain-algo none --test-model trained \
--episode 100 --use-latent --latent-dim 1 --task-transition --save-date 200219 --load-date 200219
```
#### FetchPickAndPlace-v1
```bash
python enjoy.py --env-name FetchPickAndPlace-v1 --algo ppo --gail-algo standard --pretrain-algo cvae --test-model trained \
--episode 100 --save-date 200219 --load-date 200219 --use-latent --latent-dim 4 --task-transition --seed 1
```
