"""Run the four authorized 300-episode comparisons sequentially on one GPU."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone
from train import save_json
from tune_pong import comparison

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'tuning_results'
PLAN = RESULTS / 'architecture_sweep.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_result(result, hidden, initialization):
    c = result['config']
    assert result['episodes'] == 300 and len(result['validation']) == 50
    assert c['lr'] == 0.001 and c['gamma'] == 0.99 and c['seed'] == 0
    assert c['batch_size'] == 1 and c['baseline'] == 'none'
    assert not c.get('omit_time_discount', False)
    assert c.get('hidden', 128) == hidden
    assert c.get('initialization', 'default') == initialization


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    lock = (RESULTS / 'architecture_sweep.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(PLAN.read_text())
    assert plan['status'] != 'complete', 'This bounded sweep already finished'
    assert [(r['name'], r['hidden'], r['initialization']) for r in plan['runs']] == [
        ('width32_default', 32, 'default'), ('width64_default', 64, 'default'),
        ('width256_default', 256, 'default'), ('width128_he_small', 128, 'he_small_output')]
    control = json.loads((ROOT / plan['control']['result']).read_text())
    validate_result(control, 128, 'default')
    plan.update(status='running', pid=os.getpid(), started_utc=now())
    save_json(PLAN, plan)
    child = None
    active = None

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise InterruptedError(f'Sweep interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for run in plan['runs']:
            active = run
            folder = RESULTS / run['name']
            path = folder / 'block_300.json'
            if not path.exists():
                run.update(status='running', started_utc=now())
                plan['active_name'] = run['name']
                save_json(PLAN, plan)
                command = [sys.executable, '-u', 'tune_pong.py', '--name', run['name'],
                           '--lr', '0.001', '--target', '300', '--hidden', str(run['hidden']),
                           '--initialization', run['initialization']]
                print('Starting', run['name'], flush=True)
                with (ROOT / run['log']).open('a') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    run['pid'] = child.pid
                    save_json(PLAN, plan)
                    if child.wait() != 0:
                        raise RuntimeError(f'{run["name"]} failed; see {run["log"]}')
            result = json.loads(path.read_text())
            validate_result(result, run['hidden'], run['initialization'])
            paired = comparison(result['validation'], control['validation'])
            run.update(status='complete', completed_utc=result['completed_utc'],
                       vs_128_default=paired, vs_own_initial=result['vs_initial'])
            save_json(PLAN, plan)
            v = result['vs_initial']
            lo, hi = paired['paired_bootstrap_95_interval']
            direction = ('本组得分较128默认基线更好，尚需新种子复核。' if lo > 0 else
                         '本组得分较128默认基线更差。' if hi < 0 else
                         '得分差区间跨0，当前50局不足以区分两者。')
            save_json(folder / 'stage_decision.json', {
                'time_utc': now(), 'stage': f'网络对照：{run["name"]}完成300局',
                'experiment': run['name'], 'episodes': 300,
                'result': f'50局得分{v["mean"]:.2f}，折扣回报{v["mean_discounted_return"]:.3f}；相对128默认得分差{paired["paired_score_gain"]:+.2f}，配对95%区间[{lo:+.2f},{hi:+.2f}]；动作熵{v["mean_entropy_first256"]:.3f}，概率饱和比例{100*v["mean_saturated_first256"]:.2f}%。',
                'analysis': direction + '这是单训练种子的初筛；各结构自己的未训练基准另行保存，不能把初始化差异误当作学习增益。',
                'decision': '本组达到300局即停止；按预定方案完成其余对照，全部结束后再选择。',
                'reason': '保持相同训练预算、学习率、公式和验证种子；宽度及初始化分开比较，不根据中途结果改变对照方案。'})
            subprocess.run([sys.executable, 'staged_report.py'], cwd=ROOT, check=True)
            print('Completed', run['name'], flush=True)
        plan.update(status='complete', active_name=None, completed_utc=now())
        save_json(PLAN, plan)
        print('ARCHITECTURE_SWEEP_COMPLETE', flush=True)
    except Exception as error:
        plan.update(status='failed', error=str(error), failed_utc=now())
        if active is not None and active.get('status') != 'complete':
            active.update(status='failed', error=str(error))
        save_json(PLAN, plan)
        raise


if __name__ == '__main__':
    main()
