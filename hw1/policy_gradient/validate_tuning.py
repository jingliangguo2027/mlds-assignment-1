"""Analytical checks for the loss and batch-gradient averaging used in tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from train import update, discounted_returns
from validate import exact_gradient_check


def batch_check():
    torch.manual_seed(2)
    model=nn.Linear(1,2,bias=False)
    optimizer=torch.optim.SGD(model.parameters(),lr=0)
    average=torch.zeros_like(model.weight)
    trajectories=[([0,1],[1.,2.]),([1,0],[-1.,3.])]
    for actions,rewards in trajectories:
        update(model,optimizer,np.ones((2,1),dtype=np.float32),actions,
               discounted_returns(rewards,.99),.99,0,'cpu')
        average+=model.weight.grad/2
    model.zero_grad()
    loss=0
    for actions,rewards in trajectories:
        logp=Categorical(logits=model(torch.ones(2,1))).log_prob(torch.tensor(actions))
        loss-=((.99**torch.arange(2))*torch.tensor(discounted_returns(rewards,.99))*logp).sum()/2
    loss.backward()
    error=float((average-model.weight.grad).abs().max())
    assert error<1e-7
    return error


if __name__=='__main__':
    torch.set_num_threads(1)
    result={'analytical_gradient_error':exact_gradient_check(),'batch_gradient_error':batch_check()}
    folder=Path('tuning_results'); folder.mkdir(exist_ok=True)
    (folder/'validation_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
