import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('search', 'grasp'), required=True)
    parser.add_argument('--inner-probability', type=float)
    parser.add_argument('--inner-fraction', type=float, default=0.5)
    parser.add_argument('--output', default='logs/search_grasp_1019_four')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=10000)
    args = parser.parse_args()
    if args.group == 'search' and (
        args.inner_probability is None or not 0 <= args.inner_probability <= 1
    ):
        parser.error('Search requires --inner-probability in [0, 1]')
    if not 0 < args.inner_fraction < 1:
        parser.error('--inner-fraction must be in (0, 1)')
    root = Path(__file__).resolve().parent
    output = root / args.output
    runtime = output / ('runtime_' + args.group)
    gpu_ids = (0, 1) if args.group == 'search' else (2, 3)
    usage = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'
    ], text=True)
    for row in usage.splitlines():
        gpu, memory, utilization = [int(value.strip()) for value in row.split(',')]
        if gpu in gpu_ids and (memory > 1024 or utilization > 0):
            raise RuntimeError('Requested GPU is busy: ' + row)
    runtime.mkdir(parents=True, exist_ok=False)
    for directory in ('tasks', 'algorithms', 'utils', 'cfg'):
        shutil.copytree(root / directory, runtime / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(root / 'train.py', runtime / 'train.py')
    shutil.copy2(Path(__file__), runtime / Path(__file__).name)
    (runtime / 'source.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=root, text=True))
    manifest = dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    group=args.group, runtime=str(runtime), seed=args.seed,
                    initialization='from_scratch', iterations=args.iterations, num_envs=1024,
                    rollout_steps=32, runs=[])
    manifest_path = output / (args.group + '_manifest.json')
    for gpu, experiment in zip(gpu_ids, ('E2', 'E3')):
        suffix = experiment if args.group == 'search' else (
            'E2_1015_r015_20obj' if experiment == 'E2' else 'E3_1015_exp1_resume'
        )
        cfg_env = runtime / ('cfg/shadow_hand_grasp_dexrep_' + suffix + '.yaml')
        cfg = yaml.safe_load(cfg_env.read_text())
        if args.group == 'search':
            cfg['env']['objectPositionRandomization']['innerRegionSampling'] = {
                'halfExtentFraction': args.inner_fraction, 'probability': args.inner_probability,
            }
            cfg_env.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cfg_train = runtime / ('cfg/ppo1/config_E3_1015_exp1_resume.yaml'
                               if args.group == 'grasp' and experiment == 'E3'
                               else 'cfg/ppo1/config.yaml')
        name = '{}_{}_r{}_seed{}'.format(args.group, experiment,
                                         '015' if experiment == 'E2' else '025', args.seed)
        directory = output / name
        directory.mkdir(exist_ok=False)
        command = [
            '/opt/conda/envs/Tgrasp/bin/python', '-u', str(runtime / 'train.py'),
            '--task', 'ShadowHandGraspDexRep', '--algo', 'ppo1',
            '--cfg_env', str(cfg_env), '--cfg_train', str(cfg_train),
            '--num_envs', '1024', '--seed', str(args.seed), '--max_iterations', str(args.iterations),
            '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--headless', '--logdir', str(directory),
        ]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           LD_LIBRARY_PATH='/opt/conda/envs/Tgrasp/lib',
                           PYTHONPATH=str(runtime) + ':' + str(root))
        with (directory / 'train.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        manifest['runs'].append(dict(name=name, gpu=gpu, pid=process.pid, command=command,
            directory=str(directory), experiment=experiment, episode=cfg['env']['episodeLength'],
            objects=cfg['env']['object_code_dict'], asymmetric=cfg['env']['asymmetric_observations'],
            object_randomization=cfg['env']['objectPositionRandomization']))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest['runs'][-1]), flush=True)


if __name__ == '__main__':
    main()
