import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

from evaluate_grasp import atomic_json, summarize




def gpu_available(gpu, required_memory_mib):
    output = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
        '--format=csv,noheader,nounits',
    ], text=True, timeout=15)
    if int(output.strip()) < required_memory_mib:
        return False
    processes = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-compute-apps=used_memory', '--format=csv,noheader,nounits',
    ], text=True, timeout=15)
    return all(line.strip().isdigit() and int(line.strip()) < 1024 for line in processes.splitlines())


def worker(job_path):
    job = json.loads(job_path.read_text())
    directory = Path(job['directory'])
    runtime = Path(job['runtime'])
    status_path = directory / 'status.json'

    def status(state, **details):
        atomic_json(status_path, dict(state=state, updated=time.time(), **details))

    run_lock = (directory / 'worker.lock').open('w')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = None
    status('waiting_gpu', eligible_gpus=job['gpus'])
    try:
        while gpu_lock is None:
            for gpu in job['gpus']:
                candidate = Path('/tmp/tgrasp_geometry_gpu%d.lock' % gpu).open('w')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if gpu_available(gpu, job['required_memory_mib']):
                        gpu_lock = candidate
                        break
                except BlockingIOError:
                    pass
                candidate.close()
            if gpu_lock is None:
                time.sleep(30)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           LD_LIBRARY_PATH='/opt/conda/envs/Tgrasp/lib',
                           PYTHONPATH=str(runtime) + ':' + job['root'])
        common = ['--task', job.get('task', 'ShadowHandGraspDexRep'), '--algo', 'ppo1',
                  '--cfg_env', job['cfg_env'], '--cfg_train', job['cfg_train'],
                  '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--headless']
        command = [sys.executable, '-u', str(runtime / 'train.py')] + common + [
            '--num_envs', str(job['num_envs']), '--seed', str(job['seed']),
            '--max_iterations', str(job['iterations']), '--logdir', str(directory),
        ]
        with (directory / 'train.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=job['root'], env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       pass_fds=(gpu_lock.fileno(), run_lock.fileno()))
            status('training', gpu=gpu, pid=process.pid, command=command)
            exit_code = process.wait()
        (directory / 'exit_code.txt').write_text(str(exit_code) + '\n')
        checkpoint = directory / 'checkpoint' / ('model_%d.pt' % job['iterations'])
        with (directory / 'train.log').open('rb') as stream:
            stream.seek(max(0, (directory / 'train.log').stat().st_size - 262144))
            tail = stream.read().decode(errors='replace')
        final_iteration = re.search(r'Learning iteration\s+%d/%d\b' % (
            job['iterations'] - 1, job['iterations']), tail)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not final_iteration:
            raise RuntimeError('Training incomplete, exit=%d; see train.log' % exit_code)
        if exit_code:
            status('training_exit_warning', gpu=gpu, exit_code=exit_code)
        aggregate = {}
        for mode in ('mean', 'sample'):
            rows = []
            for seed in job['eval_seeds']:
                output = directory / 'evaluation' / ('%s_seed%d' % (mode, seed))
                output.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, '-u', str(runtime / 'evaluate_grasp.py'),
                           '--runtime-root', str(runtime), '--eval-mode', mode] + common + [
                    '--model_dir', str(checkpoint), '--logdir', str(output),
                    '--num_envs', str(job['eval_envs']), '--seed', str(seed),
                ]
                atomic_json(output / 'launch.json', dict(command=command, gpu=gpu))
                with (output / 'eval.log').open('w') as stream:
                    process = subprocess.Popen(command, cwd=job['root'], env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               pass_fds=(gpu_lock.fileno(), run_lock.fileno()))
                    status('evaluating', gpu=gpu, pid=process.pid, mode=mode, seed=seed,
                           training_exit_code=exit_code)
                    try:
                        evaluation_exit = process.wait(timeout=7200)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise RuntimeError('Evaluation timeout: ' + str(output))
                if evaluation_exit:
                    raise RuntimeError('Evaluation failed: ' + str(output))
                result = json.loads((output / 'result.json').read_text())
                episodes = json.loads((output / 'episodes.json').read_text())
                if not result['complete'] or len(episodes) != job['eval_envs']:
                    raise RuntimeError('Incomplete evaluation: ' + str(output))
                rows.extend(episodes)
            aggregate[mode] = summarize(rows)
            atomic_json(directory / 'evaluation' / 'summary.json', aggregate)
        status('complete', gpu=gpu, training_exit_code=exit_code)
    except Exception as error:
        status('failed', error=str(error))
        raise
    finally:
        if gpu_lock is not None:
            gpu_lock.close()


def main():
    parser = argparse.ArgumentParser(description='Train grasp policies and automatically evaluate actor-only inference.')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', default='logs/contact_grasp_1104')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--experiments', choices=['E2', 'E3', 'ContactE2'], nargs='+', default=['ContactE2'])
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--eval-envs', type=int, default=200)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[20261102, 20261103, 20261104])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if min(args.iterations, args.num_envs, args.eval_envs) <= 0:
        parser.error('iterations and environment counts must be positive')
    root = Path(__file__).resolve().parent
    output = (root / args.output).resolve()
    runtime = output / 'runtime'
    runtime.mkdir(parents=True, exist_ok=False)
    for package in ('tasks', 'algorithms', 'utils', 'cfg'):
        shutil.copytree(root / package, runtime / package,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ('train.py', 'evaluate_grasp.py', 'launch_geometry_grasp.py'):
        shutil.copy2(root / filename, runtime / filename)
    (runtime / 'source.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=root, text=True))
    manifest = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    initialization='from_scratch', runtime=str(runtime), runs=[])
    for architecture in args.experiments:
        contact_grasp = architecture == 'ContactE2'
        cfg_env = runtime / ('cfg/shadow_hand_contact_grasp.yaml' if contact_grasp else 'cfg/shadow_hand_grasp_dexrep_' + architecture + '.yaml')
        cfg_train = runtime / ('cfg/ppo1/config_contact_grasp.yaml' if contact_grasp else ('cfg/ppo1/config.yaml' if architecture == 'E2' else 'cfg/ppo1/config_E3.yaml'))
        cfg = yaml.safe_load(cfg_env.read_text())
        tactile = cfg['env']['tactile']
        if tactile.get('search_only', False) or not cfg['env']['privileged_geometry']['enabled']:
            raise ValueError('Expected grasp configuration with privileged geometry enabled')
        if 'region_exploration' in tactile['reward'] or 'search_regions' in cfg['env']['obs_dim']:
            raise ValueError('Small-region experiments must not contain nine-region guidance')
        for seed in args.seeds:
            name = '%s_seed%d' % (architecture, seed)
            directory = output / name
            directory.mkdir(exist_ok=False)
            gpu = args.gpus[len(manifest['runs']) % len(args.gpus)]
            job = dict(name=name, root=str(root), runtime=str(runtime), directory=str(directory),
                       cfg_env=str(cfg_env), cfg_train=str(cfg_train), seed=seed,
                       task='ShadowHandContactGrasp' if contact_grasp else 'ShadowHandGraspDexRep',
                       gpus=[gpu], required_memory_mib=(10 if contact_grasp else (12 if architecture == 'E2' else 21)) * 1024,
                       iterations=args.iterations, num_envs=args.num_envs,
                       eval_envs=args.eval_envs, eval_seeds=args.eval_seeds,
                       episode_length=cfg['env']['episodeLength'], objects=cfg['env']['object_code_dict'],
                       initial_hand=tactile['initial_hand'],
                       open_hand_penalty=tactile['reward'].get('open_hand_penalty', 0.0),
                       oracle_reach_scale=cfg['env']['distRewardScale'],
                       relative_object_position=tactile['relative_object_position'],
                       voxel_exploration=tactile['reward']['voxel_exploration'],
                       palm_up_penalty_angle_degrees=tactile['reward']['palm_up_penalty_angle_degrees'],
                       local_voxel=tactile['local_voxel']['enabled'],
                       object_position_randomization=cfg['env']['objectPositionRandomization'])
            if contact_grasp:
                job['contact_grasp'] = cfg['env']['contact_grasp']
            job_path = directory / 'job.json'
            atomic_json(job_path, job)
            if not args.prepare_only:
                with (directory / 'worker.log').open('w') as stream:
                    process = subprocess.Popen([sys.executable, '-u', str(runtime / 'launch_geometry_grasp.py'),
                                                '--worker', str(job_path)], cwd=root, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                job['worker_pid'] = process.pid
            manifest['runs'].append(job)
            atomic_json(output / 'manifest.json', manifest)
            print(name, 'prepared' if args.prepare_only else 'queued', flush=True)


if __name__ == '__main__':
    main()
