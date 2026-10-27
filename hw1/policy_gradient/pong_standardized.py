"""Pong REINFORCE with standardized full-game returns."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from train import (make_policy, make_env, rollout, discounted_returns, save_json,
                   load_checkpoint, policy_fingerprint, preprocess)


def pong_env():
    return make_env('pong')


def evaluate_checkpoint(checkpoint, output, device, count=500, n_envs=8):
    """Evaluate fixed weights; resume completed games from the progress file."""
    import gymnasium as gym
    import hashlib
    import fcntl
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state = load_checkpoint(checkpoint)
    policy = make_policy('pong',128,'he_small_output',output_gain=.1).to(device)
    policy.load_state_dict(state['policy'])
    policy.eval().requires_grad_(False)
    model_id = policy_fingerprint(policy)
    training_episodes = len(state.get('records',state.get('training_scores',[])))
    assert training_episodes>0
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'evaluate.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'evaluation.json').exists():
        raise FileExistsError('Evaluation is already complete; use a new output directory')
    progress_file = output/'evaluation_progress.json'
    rows = []
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
        assert progress['model_id']==model_id and progress['target']==count
        rows = progress['rows']
    seed_start = 9000000
    completed = {r['seed']-seed_start for r in rows}
    assert len(completed)==len(rows) and all(0<=i<count for i in completed)
    remaining = [i for i in range(count) if i not in completed]
    n_envs = min(n_envs,max(1,len(remaining)))
    torch.set_num_threads(1)
    envs = gym.vector.AsyncVectorEnv([pong_env]*n_envs,context='spawn',
                                     autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    active = remaining[:n_envs] if remaining else [None]
    pending = iter(remaining[n_envs:])
    raw,_ = envs.reset(seed=[seed_start+i if i is not None else None for i in active])
    previous = np.zeros((n_envs,80,80),dtype=np.float32)
    rngs = [np.random.default_rng(seed_start+(i or 0)+100000) for i in active]
    scores = np.zeros(n_envs)
    steps = np.zeros(n_envs,dtype=np.int64)
    try:
        while len(rows)<count:
            current = np.stack([preprocess(obs) for obs in raw])
            states = (current-previous).reshape(n_envs,-1)
            previous = current
            with torch.no_grad():
                p0 = policy(torch.as_tensor(states,device=device)).softmax(-1)[:,0].cpu().numpy()
            actions = np.array([2 if r.random()<p else 3 for r,p in zip(rngs,p0)])
            raw,rewards,terminated,truncated,_ = envs.step(actions)
            scores += rewards
            steps += 1
            done = terminated|truncated
            reset_seeds = [None]*n_envs
            for i in np.flatnonzero(done):
                ep = active[i]
                if ep is not None:
                    rows.append(dict(seed=seed_start+ep,action_seed=seed_start+ep+100000,
                                     score=float(scores[i]),steps=int(steps[i])))
                next_episode = next(pending,None)
                if next_episode is not None:
                    active[i] = next_episode
                    reset_seeds[i] = seed_start+next_episode
                    rngs[i] = np.random.default_rng(seed_start+next_episode+100000)
                else:
                    active[i] = None
                scores[i],steps[i] = 0.,0
                previous[i] = 0.
            if done.any():
                save_json(output/'evaluation_progress.json',dict(completed=len(rows),target=count,
                          model_id=model_id,rows=sorted(rows,key=lambda r:r['seed'])))
                print(f'Evaluate {len(rows)}/{count}: mean={np.mean([r["score"] for r in rows]):.2f}',flush=True)
                if len(rows)<count:
                    raw,_ = envs.reset(seed=reset_seeds,options={'reset_mask':done})
    finally:
        envs.close()
    assert len(rows)==count and len({r['seed'] for r in rows})==count
    assert model_id==policy_fingerprint(policy)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_hash
    values = np.array([r['score'] for r in rows])
    result = dict(training_episodes=training_episodes,test_episodes=count,
                  test_mean=float(values.mean()),test_std=float(values.std(ddof=0)),
                  test_wins=int((values>0).sum()),test_win_rate=float((values>0).mean()),
                  test_scores=[r['score'] for r in sorted(rows,key=lambda r:r['seed'])],
                  rows=sorted(rows,key=lambda r:r['seed']),source_checkpoint_sha256=source_hash,
                  model_id=model_id,policy_unchanged=True,stochastic_actions=True,
                  sampling='Independent per-game NumPy RNG; sample from network probabilities',
                  config=state['config'])
    save_json(output/'evaluation.json',result)
    plot_evaluation(output,result)


def episode_gradient(policy, states, actions, returns, baseline, device):
    """Accumulate the standardized full-episode loss, without an outer discount."""
    policy.zero_grad()
    a = torch.as_tensor(returns-baseline, device=device)
    a = (a-a.mean())/(a.std(correction=1)+1e-8)
    total = 0.
    for start in range(0,len(actions),256):
        end = start+256
        x = torch.as_tensor(states[start:end],dtype=torch.float32,device=device)
        acts = torch.as_tensor(actions[start:end],dtype=torch.long,device=device)
        loss = -(Categorical(logits=policy(x)).log_prob(acts)*a[start:end]).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite policy loss')
        loss.backward()
        total += loss.item()
    return total


def plot_evaluation(output, test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'axes.spines.top':False,'axes.spines.right':False,'font.size':11}):
        fig, ax = plt.subplots(figsize=(8.4,2.7),layout='constrained')
        ax.hist(test['test_scores'],bins=np.arange(-21.5,22.5,1),
                color='#417da1',edgecolor='white')
        ax.set(xlabel='Episode reward',ylabel='Count',xlim=(-22,22))
        ax.text(.03,.95,f'Mean {test["test_mean"]:.2f}\nSD {test["test_std"]:.2f}\nWins {test["test_win_rate"]:.1%}',
                transform=ax.transAxes,va='top')
        fig.savefig(output/'evaluation_rewards.png',dpi=180)
        plt.close(fig)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = json.loads((output/'metrics.json').read_text())
    scores = np.asarray(m['training_scores'])
    assert len(scores)==m['training_episodes'] and len(scores)>=100
    x = np.arange(1,len(scores)+1)
    average = np.convolve(scores,np.ones(100)/100,mode='valid')
    wins = np.convolve((scores>0).astype(float),np.ones(100),mode='valid')
    style = {'axes.spines.top':False,'axes.spines.right':False,'font.size':11}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.4,3.45),layout='constrained')
        ax.plot(x,scores,color='#9cb9cf',alpha=.35,lw=.6,label='Episode reward')
        ax.plot(x[99:],average,color='#c76022',lw=2,label='100-episode average')
        ax.axhline(0,color='#777777',lw=.7,ls='--')
        ax.set(xlabel='Training episode',ylabel='Episode reward',xlim=(0,len(scores)),ylim=(-22,22))
        ax.legend(loc='upper left',frameon=False)
        fig.savefig(output/'training_rewards.png',dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.4,3.05),layout='constrained')
        ax.plot(x[99:],wins,color='#176a6d',lw=2)
        ax.axhline(50,color='#888888',lw=.7,ls='--')
        ax.scatter([len(scores)],[wins[-1]],color='#176a6d',s=20)
        ax.annotate(f'{wins[-1]:.0f}%',(len(scores),wins[-1]),xytext=(-35,9),textcoords='offset points')
        ax.set(xlabel='Training episode',ylabel='Win rate: last 100 games (%)',xlim=(0,len(scores)),ylim=(-2,103))
        fig.savefig(output/'training_win_rate.png',dpi=180)
        plt.close(fig)
        test_file = output/'evaluation.json'
        if test_file.exists():
            plot_evaluation(output,json.loads(test_file.read_text()))
        elif m.get('test_scores'):
            plot_evaluation(output,m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot-only',action='store_true',help='Redraw saved results without training')
    p.add_argument('--episodes',type=int,default=4000,help='Training games; divisible by 10')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--eval-episodes',type=int,default=0,help='Number of frozen-policy test games')
    p.add_argument('--evaluate-checkpoint',type=Path,help='Test saved weights instead of training')
    p.add_argument('--vector-envs',type=int,default=8,help='Parallel CPU environments for checkpoint tests')
    args = p.parse_args()
    if args.plot_only:
        if (args.output/'metrics.json').exists():
            plot(args.output)
        else:
            plot_evaluation(args.output,json.loads((args.output/'evaluation.json').read_text()))
        return
    if args.evaluate_checkpoint:
        if not 1<=args.vector_envs<=16 or args.eval_episodes<args.vector_envs:
            p.error('Use 1-16 vector environments and at least that many evaluation games')
        if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
            p.error('Expose exactly one GPU')
        evaluate_checkpoint(args.evaluate_checkpoint,args.output,args.device,args.eval_episodes,args.vector_envs)
        return
    if args.episodes<100 or args.episodes%10 or args.eval_episodes<0:
        p.error('Use at least 100 training episodes, divisible by 10; evaluation count must be nonnegative')
    if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        p.error('Expose exactly one GPU')
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError('Use an empty output directory')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = make_policy('pong',128,'he_small_output',output_gain=.1).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(),lr=.001)
    scores, baseline = [], 0.
    config = dict(gamma=.99,lr=.001,batch_size=10,hidden=128,seed=0,output_gain=.1,
                  initialization='he_small_output',normalize_advantages=True,
                  normalization_std_correction=1,omit_time_discount=True,
                  reset_returns_on_point=False,baseline='ema',ema_decay=.9)
    env = make_env('pong')
    try:
        for start in range(0,args.episodes,10):
            gradients = [torch.zeros_like(v) for v in policy.parameters()]
            references = []
            for ep in range(start,start+10):
                torch.manual_seed(10000+ep)
                states,actions,rewards,_ = rollout(env,policy,'pong',args.device,ep)
                returns = discounted_returns(rewards,.99)
                episode_gradient(policy,states,actions,returns,baseline,args.device)
                for g,v in zip(gradients,policy.parameters()):
                    g.add_(v.grad/10)
                references.append(float(np.average(returns,weights=.99**np.arange(len(returns)))))
                scores.append(sum(rewards))
            for v,g in zip(policy.parameters(),gradients):
                assert torch.isfinite(g).all()
                v.grad = g
            optimizer.step()
            assert all(torch.isfinite(v).all() for v in policy.parameters())
            baseline = .9*baseline+.1*float(np.mean(references))
            tmp = args.output/'checkpoint.tmp'
            torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),
                            training_scores=scores,baseline=baseline,config=config),tmp)
            tmp.replace(args.output/'checkpoint.pt')
            print(f'Train {len(scores)}: last100={np.mean(scores[-100:]):.2f}',flush=True)
    finally:
        env.close()
    m = dict(config=config,training_scores=scores,training_episodes=len(scores),
             last100_mean=float(np.mean(scores[-100:])),
             last100_win_rate=float(np.mean(np.asarray(scores[-100:])>0)),
             endpoint_checkpoint_available=True,independent_test_episodes=args.eval_episodes)
    if args.eval_episodes:
        env = make_env('pong')
        test = []
        try:
            for seed in range(1000000,1000000+args.eval_episodes):
                torch.manual_seed(seed)
                _,_,r,_ = rollout(env,policy,'pong',args.device,seed)
                test.append(sum(r))
        finally:
            env.close()
        m.update(test_scores=test,test_mean=float(np.mean(test)),test_std=float(np.std(test)),
                 test_win_rate=float(np.mean(np.asarray(test)>0)))
    save_json(args.output/'metrics.json',m)
    plot(args.output)


if __name__=='__main__':
    main()
