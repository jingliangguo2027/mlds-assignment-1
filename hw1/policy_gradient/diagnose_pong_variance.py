"""Paired gradient-variance diagnostic under one frozen Pong policy."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from train import make_policy,make_env,rollout,discounted_returns,update

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('tuning_results'))
output=parser.parse_args().output
output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
torch.manual_seed(0)
policy=make_policy('pong').cuda()
optimizer=torch.optim.SGD(policy.parameters(),lr=0)
env=make_env('pong')
b=0.0
for episode in range(20):
    seed=300000+episode
    torch.manual_seed(seed)
    _,_,rewards,_=rollout(env,policy,'pong','cuda',seed)
    returns=discounted_returns(rewards,.99)
    b=.9*b+.1*float(np.average(returns,weights=.99**np.arange(len(returns))))
print('Independent pilot baseline:',b,flush=True)
samples={'none':[],'historical_ema':[]}
for episode in range(20):
    seed=310000+episode
    torch.manual_seed(seed)
    states,actions,rewards,_=rollout(env,policy,'pong','cuda',seed)
    returns=discounted_returns(rewards,.99)
    for name,baseline in [('none',0.0),('historical_ema',b)]:
        update(policy,optimizer,states,actions,returns,.99,baseline,'cuda')
        samples[name].append(torch.cat([p.grad.flatten() for p in policy.parameters()]).detach().cpu())
    if (episode+1)%10==0:
        print('Paired gradient episodes:',episode+1,flush=True)
env.close()
result={'pilot_episodes':20,'paired_episodes':20,'baseline':b,'statistics':{}}
for name,gradients in samples.items():
    g=torch.stack(gradients)
    variance=float(g.var(0,unbiased=True).sum())
    signal=float(g.mean(0).norm())
    result['statistics'][name]={'sum_component_variances':variance,'sample_mean_gradient_norm':signal,
                                'sample_mean_norm_over_gradient_std':signal/variance**.5}
result['variance_ratio']=result['statistics']['historical_ema']['sum_component_variances']/result['statistics']['none']['sum_component_variances']
(output/'gradient_variance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
