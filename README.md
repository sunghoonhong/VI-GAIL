# Video Input Generative Adversarial Imitation Learning

Stablizing method of training GAIL for Video input

# Usage

- expert.py  
Simulate Expert Trajectory  
To gather expert demonstrations run this,
> python expert.py --record --episode=[EPISODE_NUM]
- bc_pretrain.py  
Behavior Cloning Pretrain for Video Encoder
- code_pretrain.py  
Code Posterior Pretrain for DI-GAIL variants
- gail.py  
(main) Train VI-GAIL

