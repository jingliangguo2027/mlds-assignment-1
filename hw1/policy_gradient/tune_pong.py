"""Controlled Pong experiments, including the optional removal of outer gamma**t."""
import argparse
import json
from pathlib import Path
import time
import shutil
from datetime import datetime, timezone
import numpy as np
import torch
from torch.distributions import Categorical
from train import (make_policy, make_env, rollout, discounted_returns, update,
                   save_json, load_checkpoint, policy_fingerprint)

GAMMA = 0.99
DEVICE = 'cuda'


def vector(policy):
    return torch.cat([p.detach().reshape(-1) for p in policy.parameters()])


def evaluate(policy, count=20, seed_start=200000):
    """Tuning validation only. Never use the final test seeds (1000000+)."""
    env = make_env('pong')
    rows = []
    model_id = policy_fingerprint(policy)
    with torch.no_grad():
        for i in range(count):
            seed = seed_start + i
            torch.manual_seed(seed)
            states, actions, rewards, _ = rollout(env, policy, 'pong', DEVICE, seed)
            probs = policy(torch.as_tensor(states[:256], dtype=torch.float32, device=DEVICE)).softmax(-1)
            rows.append({'seed': seed, 'score': sum(rewards),
                         'first_reward_step': next((j for j,r in enumerate(rewards) if r), None),
                         'points_won': rewards.count(1.0), 'score_first300': sum(rewards[:300]),
                         'discounted_return': float(discounted_returns(rewards, GAMMA)[0]),
                         'steps': len(rewards), 'right_fraction': float(np.mean(np.array(actions)==0)),
                         'entropy_first256': float(Categorical(probs=probs).entropy().mean()),
                         'saturated_first256': float((probs.max(-1).values > .99).float().mean())})
    env.close()
    assert model_id == policy_fingerprint(policy), 'Evaluation changed the policy'
    return rows


def comparison(rows, reference):
    assert [r['seed'] for r in rows] == [r['seed'] for r in reference]
    delta = np.array([r['score']-b['score'] for r,b in zip(rows,reference)])
    rng = np.random.default_rng(42)
    boot = rng.choice(delta, size=(5000,len(delta)), replace=True).mean(1)
    low, high = np.quantile(boot,[.025,.975])
    jdelta=np.array([r['discounted_return']-b['discounted_return'] for r,b in zip(rows,reference)])
    jboot=rng.choice(jdelta,size=(5000,len(jdelta)),replace=True).mean(1)
    return {'mean': float(np.mean([r['score'] for r in rows])),
            'paired_episodes':len(rows),'discounted_gain':float(jdelta.mean()),
            'discounted_gain_95_interval':np.quantile(jboot,[.025,.975]).tolist(),
            'mean_discounted_return': float(np.mean([r['discounted_return'] for r in rows])),
            'paired_score_gain': float(delta.mean()), 'paired_bootstrap_95_interval': [float(low),float(high)],
            'clear_gain': bool(delta.mean() >= 1.0 and low > 0),
            'mean_entropy_first256': float(np.mean([r['entropy_first256'] for r in rows])),
            'mean_saturated_first256': float(np.mean([r['saturated_first256'] for r in rows]))}


def audit(output, old_checkpoint):
    torch.manual_seed(0)
    policy = make_policy('pong').to(DEVICE)
    env = make_env('pong')
    torch.manual_seed(123)
    states, actions, rewards, _ = rollout(env, policy, 'pong', DEVICE, 123)
    env.close()
    returns = discounted_returns(rewards,GAMMA)
    no_step = torch.optim.SGD(policy.parameters(),lr=0)
    chunk_loss = update(policy,no_step,states,actions,returns,GAMMA,0,DEVICE)
    chunk_grad = torch.cat([p.grad.flatten() for p in policy.parameters()]).detach().clone()
    policy.zero_grad()
    logp = Categorical(logits=policy(torch.as_tensor(states,dtype=torch.float32,device=DEVICE))).log_prob(torch.tensor(actions,device=DEVICE))
    loss = -(logp*torch.tensor(returns,device=DEVICE)*GAMMA**torch.arange(len(actions),device=DEVICE)).sum()
    loss.backward()
    full_grad = torch.cat([p.grad.flatten() for p in policy.parameters()]).detach().clone()
    relative_error = float((chunk_grad-full_grad).norm()/full_grad.norm())
    assert relative_error < 1e-4 and torch.isfinite(full_grad).all() and full_grad.norm()>0
    before = float(loss.detach())
    # Fixed-trajectory directional check: a small descent step must lower its surrogate loss.
    originals = [p.detach().clone() for p in policy.parameters()]
    with torch.no_grad():
        for p in policy.parameters():
            p -= 1e-5*p.grad/full_grad.norm()
        after = float(-(Categorical(logits=policy(torch.as_tensor(states,dtype=torch.float32,device=DEVICE))).log_prob(torch.tensor(actions,device=DEVICE))*torch.tensor(returns,device=DEVICE)*GAMMA**torch.arange(len(actions),device=DEVICE)).sum())
        for p, original in zip(policy.parameters(), originals):
            p.copy_(original)
    assert after < before
    initial = evaluate(policy)
    result = {'chunk_vs_full_gradient_relative_error':relative_error,'gradient_norm':float(full_grad.norm()),
              'surrogate_before':before,'surrogate_after_descent':after,
              'input_nonzero_fraction':float(np.mean(states != 0)), 'episode_steps':len(actions),
              'late_weight_1000':GAMMA**1000,'initial_validation':initial}
    if old_checkpoint:
        checkpoint = load_checkpoint(old_checkpoint)
        policy.load_state_dict(checkpoint['policy'])
        old = evaluate(policy)
        result.update({'old_training_episodes':len(checkpoint['history']), 'old_validation':old,
                       'old_vs_initial':comparison(old,initial)})
    save_json(output/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2),flush=True)


def train_block(output, name, lr, batch_size, baseline, target, max_block=300, eval_count=50,
                omit_time_discount=False, hidden=128, initialization='default'):
    folder = output/name
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'checkpoint.pt'
    torch.manual_seed(0)
    policy = make_policy('pong', hidden, initialization).to(DEVICE)
    initial_fingerprint = policy_fingerprint(policy)
    optimizer = torch.optim.Adam(policy.parameters(),lr=lr)
    no_step = torch.optim.SGD(policy.parameters(),lr=0)
    config = {'lr':lr,'batch_size':batch_size,'baseline':baseline,'gamma':GAMMA,'seed':0,
              'hidden':hidden,'initialization':initialization,
              'omit_time_discount':omit_time_discount,
              'objective':('-sum((G_t-b) * log_prob), averaged across batch episodes; G remains discounted'
                           if omit_time_discount else '-sum(gamma**t * (G_t-b) * log_prob), averaged across batch episodes'),
              'training_env_seeds':'0,1,...','training_action_seeds':'10000,10001,...',
              'validation_seeds':f'200000..{200000+eval_count-1}','device_name':torch.cuda.get_device_name(0),
              'torch':str(torch.__version__)}
    records,updates,b = [],[],0.0
    if path.exists():
        checkpoint=load_checkpoint(path)
        defaults={'omit_time_discount':False,'hidden':128,'initialization':'default'}
        assert all(checkpoint['config'].get(k,defaults.get(k))==v
                   for k,v in config.items() if k!='validation_seeds'), 'Use a new experiment name when changing training settings'
        policy.load_state_dict(checkpoint['policy'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        records,updates,b=checkpoint['records'],checkpoint['updates'],checkpoint['baseline_value']
    start_episode=len(records)
    assert target>start_episode and target-start_episode<=max_block and target%batch_size==0
    if path.exists() and not (folder/f'checkpoint_{start_episode}.pt').exists():
        shutil.copy2(path,folder/f'checkpoint_{start_episode}.pt')
    save_json(folder/'config.json',config)
    if not (folder/'initialization.json').exists():
        torch.manual_seed(0)
        initial = make_policy('pong',hidden,initialization)
        save_json(folder/'initialization.json',{
            'hidden':hidden,'initialization':initialization,'seed':0,
            'model_fingerprint':initial_fingerprint,
            'parameter_count':sum(p.numel() for p in initial.parameters()),
            'parameters':{k:{'mean':float(v.mean()),'std':float(v.std()),
                             'min':float(v.min()),'max':float(v.max())}
                          for k,v in initial.state_dict().items()}})
    env=make_env('pong')
    started=time.time()
    for start in range(len(records),target,batch_size):
        sums=[torch.zeros_like(p) for p in policy.parameters()]
        references=[]
        before=vector(policy).clone()
        probe=None
        for episode in range(start,start+batch_size):
            torch.manual_seed(10000+episode)
            states,actions,rewards,_=rollout(env,policy,'pong',DEVICE,episode)
            returns=discounted_returns(rewards,GAMMA)
            used=b if baseline=='ema' else 0.0
            loss=update(policy,no_step,states,actions,returns,GAMMA,used,DEVICE,
                        omit_time_discount=omit_time_discount)
            for total,p in zip(sums,policy.parameters()):
                assert torch.isfinite(p.grad).all()
                total.add_(p.grad/batch_size)
            references.append(float(np.average(returns,weights=GAMMA**np.arange(len(returns)))))
            records.append({'episode':episode+1,'score':sum(rewards),'discounted_return':float(returns[0]),
                            'steps':len(actions),'loss':loss,'baseline_used':used})
            if probe is None:
                probe=torch.as_tensor(states[:256],dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            old_probs=policy(probe).softmax(-1)
        for p,total in zip(policy.parameters(),sums):
            p.grad=total
        grad_norm=float(torch.cat([p.grad.flatten() for p in policy.parameters()]).norm())
        optimizer.step()
        assert torch.isfinite(vector(policy)).all()
        with torch.no_grad():
            new_probs=policy(probe).softmax(-1)
            kl=float((old_probs*(old_probs.clamp_min(1e-12).log()-new_probs.clamp_min(1e-12).log())).sum(-1).mean())
            delta=float((vector(policy)-before).norm())
        updates.append({'episode':len(records),'gradient_norm':grad_norm,'parameter_delta_norm':delta,
                        'relative_parameter_delta':delta/float(before.norm()),'probe_kl':kl,
                        'probe_entropy':float(Categorical(probs=new_probs).entropy().mean()),
                        'probe_saturated_fraction':float((new_probs.max(-1).values>.99).float().mean())})
        if baseline=='ema':
            b=.9*b+.1*float(np.mean(references))
        if len(records)%10==0 or len(records)==target:
            temporary=path.with_suffix('.tmp')
            torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),
                        'records':records,'updates':updates,'baseline_value':b,'config':config},temporary)
            temporary.replace(path)
            print(f'{name} {len(records)}/{target} mean10={np.mean([r["score"] for r in records[-10:]]):.2f}',flush=True)
    env.close()
    shutil.copy2(path,folder/f'checkpoint_{target}.pt')
    print(f'{name}: validating {eval_count} episodes',flush=True)
    rows=evaluate(policy,eval_count)
    reference_path=output/(f'initial_validation_{eval_count}.json'
                           if hidden==128 and initialization=='default' else
                           f'initial_validation_h{hidden}_{initialization}_{eval_count}.json')
    if reference_path.exists():
        reference=json.loads(reference_path.read_text())
    else:
        torch.manual_seed(0)
        reference=evaluate(make_policy('pong',hidden,initialization).to(DEVICE),eval_count)
        save_json(reference_path,reference)
    previous_paths=[p for p in folder.glob('block_*.json') if int(p.stem.split('_')[1])<target]
    prior=max(previous_paths,key=lambda p:int(p.stem.split('_')[1])) if previous_paths else None
    if prior is None and (folder/'validation_at_fork.json').exists():
        prior=folder/'validation_at_fork.json'
    previous=json.loads(prior.read_text())['validation'] if prior else None
    seeds={r['seed'] for r in previous} if previous else set()
    against_previous=comparison([r for r in rows if r['seed'] in seeds],previous) if previous else None
    result={'config':config,'episodes':target,'model_fingerprint':policy_fingerprint(policy),
            'initial_reference':str(reference_path),'initial_model_fingerprint':initial_fingerprint,
            'validation':rows,'vs_initial':comparison(rows,reference),
            'vs_previous_block':against_previous,'training_first50':float(np.mean([r['score'] for r in records[:50]])),
            'training_last50':float(np.mean([r['score'] for r in records[-50:]])),
            'start_episode':start_episode,'new_training_episodes':target-start_episode,
            'previous_checkpoint_episodes':json.loads(prior.read_text())['episodes'] if prior else 0,
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'updates_last_block':updates[-max(1,(target-start_episode)//batch_size):], 'elapsed_seconds':time.time()-started}
    if (folder/'lineage.json').exists():
        result['lineage']=json.loads((folder/'lineage.json').read_text())
    gain=against_previous or result['vs_initial']
    result['decision']='stage complete; review score, discounted objective and exploration before continuing'
    result['signal']='score_gain' if gain['clear_gain'] else ('collapse' if result['vs_initial']['mean_saturated_first256']>.5 else 'needs_trend_review')
    save_json(folder/f'block_{target}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('validation','updates_last_block','config')},indent=2),flush=True)
    write_log(output)


def write_log(output):
    lines=['# Pong 小规模实验记录','','旧实验使用外层gamma**t；用户随后要求的新对照通过 --omit-time-discount 去掉该权重，G内部折扣不变。默认每300局检查一次，可按证据调整。',
           '从零训练使用相同初始化和训练种子；学习率分支显式记录继承模型与Adam状态。首轮20局验证，后续默认50局；相邻阶段只比较双方共有种子。验证用于调参，不能当作最终500局测试。',
           '首轮100局硬停止规则已撤销。现在综合得分、折扣目标、探索和多个阶段趋势判断，具体决定另记于阶段决策记录。',
           '这是单训练种子的小规模筛选，区间只反映验证局波动，多次筛选可能过拟合验证集；入选配置仍需新种子确认。','',
           '| 实验 | 外层折扣 | 局数 | 学习率 | 每次更新局数 | 验证得分 | 折扣回报 | 相对初始得分增益 | 分析/决定 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for path in sorted(output.glob('*/block_*.json')):
        r=json.loads(path.read_text()); c=r['config']; v=r['vs_initial']
        decision=r['decision']
        outer='去掉' if c.get('omit_time_discount',False) else '保留'
        lines.append(f'| {path.parent.name} | {outer} | {r["episodes"]} | {c["lr"]} | {c["batch_size"]} | {v["mean"]:.2f} | {v["mean_discounted_return"]:.3f} | {v["paired_score_gain"]:+.2f} | {decision} |')
    (output/'实验记录.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('tuning_results'))
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--old-checkpoint',type=Path)
    parser.add_argument('--name',default='lr_0001')
    parser.add_argument('--lr',type=float,default=.0001)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--baseline',choices=['none','ema'],default='none')
    parser.add_argument('--target',type=int,default=300)
    parser.add_argument('--max-block',type=int,default=300)
    parser.add_argument('--eval-episodes',type=int,default=50)
    parser.add_argument('--omit-time-discount',action='store_true')
    parser.add_argument('--hidden',type=int,default=128)
    parser.add_argument('--initialization',choices=['default','he_small_output'],default='default')
    args=parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.audit:
        assert not args.omit_time_discount, 'The original objective audit uses outer discounting'
        audit(args.output,args.old_checkpoint)
    else:
        train_block(args.output,args.name,args.lr,args.batch_size,args.baseline,args.target,
                    args.max_block,args.eval_episodes,args.omit_time_discount,args.hidden,args.initialization)
