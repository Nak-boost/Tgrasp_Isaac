import argparse
import csv
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def summarize(rows):
    count = len(rows)
    if not count:
        return {'episodes': 0}
    result = {'episodes': count}
    for key in ('contact', 'valid_grasp', 'lifted', 'stable_lift', 'far_touch',
                'first_touch_all_far', 'object_below_table', 'loss_timeout', 'timeout',
                'workspace_failure', 'terminal_success'):
        rate = sum(row[key] for row in rows) / count
        denominator = 1 + 1.96 ** 2 / count
        center = (rate + 1.96 ** 2 / (2 * count)) / denominator
        width = 1.96 * (rate * (1 - rate) / count + 1.96 ** 2 / (4 * count ** 2)) ** 0.5 / denominator
        result[key + '_rate'] = rate
        result[key + '_ci95'] = [max(0, center - width), min(1, center + width)]
    first_steps = sorted(row['first_contact_step'] for row in rows if row['contact'])
    for label, fraction in (('median', 0.5), ('p90', 0.9)):
        if first_steps:
            index = (len(first_steps) - 1) * fraction
            lower = int(index)
            result['first_contact_' + label] = first_steps[lower] + (first_steps[min(lower + 1, len(first_steps) - 1)] - first_steps[lower]) * (index - lower)
        else:
            result['first_contact_' + label] = None
    result['first_contact_mean'] = sum(first_steps) / len(first_steps) if first_steps else None
    result['regions_mean'] = sum(row['regions'] for row in rows) / count
    result['regions_histogram'] = [sum(row['regions'] == region for row in rows) for region in range(10)]
    result['table_displacement_max_m'] = max(row['table_displacement_max_m'] for row in rows)
    result['by_object'] = {}
    for name in sorted({row['object_code'] for row in rows}):
        selected = [row for row in rows if row['object_code'] == name]
        result['by_object'][name] = {
            'episodes': len(selected),
            'contact_rate': sum(row['contact'] for row in selected) / len(selected),
            'stable_lift_rate': sum(row['stable_lift'] for row in selected) / len(selected),
        }
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--eval-mode', choices=['sample', 'mean'], required=True)
    parser.add_argument('--runtime-root', required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    sys.path[:0] = [str(Path(options.runtime_root).resolve()), os.getcwd()]
    from utils.config import get_args, load_cfg, parse_sim_params, set_seed
    from utils.parse_task import parse_task
    from utils.process_marl import get_AgentIndex
    from algorithms.rl.ppo1.multimodal_policy import ActorCriticMultimodal
    import numpy as np
    import torch

    torch.set_num_threads(2)
    args = get_args()
    output = Path(args.logdir)
    output.mkdir(parents=True, exist_ok=True)
    cfg, train_cfg, _ = load_cfg(args)
    cfg['env']['random_time'] = False
    set_seed(args.seed)
    task, env = parse_task(args, cfg, train_cfg, parse_sim_params(args, cfg, train_cfg), get_AgentIndex(cfg))
    policy = ActorCriticMultimodal(
        env.observation_space.shape, env.action_space.shape,
        train_cfg['learn']['init_noise_std'], train_cfg['policy'], train_cfg['encoder'], cfg['env'],
    ).to(task.device)
    policy.load_state_dict(torch.load(args.model_dir, map_location=task.device), strict=True)
    policy.eval()

    pad_radius_by_name = {}
    for body in ET.parse('../assets/mjcf/open_ai_assets/hand/robot_tactile.xml').getroot().iter('body'):
        if body.get('name', '').endswith('_sensor'):
            geom = body.find('geom')
            size = np.fromstring(geom.get('size'), sep=' ')
            offset = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
            pad_radius_by_name[body.get('name')] = float(np.linalg.norm(size) + np.linalg.norm(offset))
    object_radii = []
    for name in task.object_code_list:
        vertices = []
        for mesh in Path('../assets/grab_meshdata', 'grab-' + name, 'coacd').glob('coacd_convex_piece_*.obj'):
            vertices.extend([float(value) for value in line.split()[1:4]] for line in mesh.read_text().splitlines() if line.startswith('v '))
        if not vertices:
            raise ValueError('Missing collision mesh: ' + name)
        object_radii.append(float(np.linalg.norm(vertices, axis=1).max()))
    object_radius = torch.tensor([object_radii[index % len(object_radii)] for index in range(task.num_envs)], device=task.device)
    pad_radii = torch.tensor([pad_radius_by_name[name] for name in task.touch_sensor_bodies], device=task.device)
    set_seed(args.seed)
    observations = env.reset()
    hidden = policy.initial_recurrent_state(task.num_envs, task.device)
    initial_positions = task.object_pos.detach().cpu().tolist()
    table_start = task.root_state_tensor[task.table_indices, :3].clone()
    active = torch.ones(task.num_envs, device=task.device, dtype=torch.bool)
    flags = {key: torch.zeros_like(active) for key in ('valid_grasp', 'stable_lift', 'far_touch', 'first_touch_all_far', 'object_below_table')}
    seen_touch = torch.zeros_like(active)
    table_displacement = torch.zeros(task.num_envs, device=task.device)
    returns = torch.zeros_like(table_displacement)
    rows = []
    started = time.monotonic()

    def record(done, info):
        nonlocal seen_touch, table_displacement
        touch = task.binary_touch > 0.5
        distance = torch.linalg.vector_norm(task.touch_sensor_pos - task.object_pos[:, None, :], dim=-1)
        far = distance > object_radius[:, None] + pad_radii[None, :] + 0.02
        first_touch = touch.any(dim=1) & ~seen_touch
        flags['valid_grasp'] |= active & task.valid_grasp_mask.bool()
        flags['stable_lift'] |= active & (task.consecutive_lift_hold_steps >= task.touch_lift_hold_steps)
        flags['far_touch'] |= active & (touch & far).any(dim=1)
        flags['first_touch_all_far'] |= active & first_touch & ~(touch & ~far).any(dim=1)
        flags['object_below_table'] |= active & (task.object_pos[:, 2] < task.table_top_z - 0.02)
        seen_touch |= touch.any(dim=1)
        displacement = torch.linalg.vector_norm(task.root_state_tensor[task.table_indices, :3] - table_start, dim=-1)
        table_displacement = torch.where(active, torch.maximum(table_displacement, displacement), table_displacement)
        for index in (active & done.bool()).nonzero().flatten().tolist():
            contact = bool(task.episode_had_contact[index])
            row = {
                'env': index, 'object_code': task.object_code_list[index % len(task.object_code_list)],
                'initial_x': initial_positions[index][0], 'initial_y': initial_positions[index][1],
                'initial_z': initial_positions[index][2], 'contact': contact,
                'first_contact_step': float(task.episode_first_contact_step[index]) if contact else None,
                'length': int(task.episode_elapsed_steps[index]),
                'regions': int(task.region_exploration.explored_regions[index].sum()),
                'lifted': bool(task.episode_had_lift[index]),
                'loss_timeout': bool(info['contact_loss_timeout'][index]) if 'contact_loss_timeout' in info else False,
                'timeout': bool(task.progress_buf[index] >= task.max_episode_length),
                'workspace_failure': bool(info['workspace_failure'][index]) if 'workspace_failure' in info else False,
                'terminal_success': bool(task.successes[index]),
                'return': float(returns[index]),
                'table_displacement_max_m': float(table_displacement[index]),
            }
            if cfg['env'].get('tactile', {}).get('search_only', False) and contact:
                sensor_indices = touch[index].nonzero().flatten()
                row['contact_sensor_names'] = [task.touch_sensor_bodies[sensor] for sensor in sensor_indices.tolist()]
                row['contact_sensor_positions'] = task.touch_sensor_pos[index, sensor_indices].cpu().tolist()
                row['contact_sensor_forces'] = task.touch_force[index, sensor_indices].cpu().tolist()
                row['contact_object_position'] = task.object_pos[index].cpu().tolist()
                row['contact_table_pose'] = task.root_state_tensor[task.table_indices[index], :7].cpu().tolist()
            row.update({key: bool(value[index]) for key, value in flags.items()})
            rows.append(row)
        active.logical_and_(~done.bool())

    with torch.no_grad():
        returns += task.rew_buf
        record(task.reset_buf, task.extras)
        for step in range(task.max_episode_length + 2):
            if not active.any():
                break
            features, next_hidden = policy.apply_recurrent_memory(policy.encode_observations(observations), hidden)
            mean, log_std = policy.action_parameters(features)
            actions = policy.action_distribution(mean, log_std).sample() if options.eval_mode == 'sample' else mean
            if not torch.isfinite(actions).all():
                raise ValueError('Non-finite actor action')
            observations, reward, done, info = env.step(actions)
            if next_hidden is not None:
                hidden = next_hidden * (~done.bool()).reshape(-1, 1).to(next_hidden.dtype)
            returns += reward * active
            record(done, info)
            if (step + 1) % 200 == 0:
                print('EVAL', step + 1, 'completed', len(rows), '/', task.num_envs, flush=True)
    if active.any():
        raise RuntimeError('Incomplete first-cohort episodes; refusing biased summary')
    rows.sort(key=lambda row: row['env'])
    atomic_json(output / 'episodes.json', rows)
    with (output / 'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        'checkpoint': str(Path(args.model_dir).resolve()),
        'checkpoint_sha256': hashlib.sha256(Path(args.model_dir).read_bytes()).hexdigest(),
        'env_config': args.cfg_env, 'train_config': args.cfg_train, 'runtime_root': options.runtime_root,
        'seed': args.seed, 'mode': options.eval_mode, 'num_envs': task.num_envs,
        'episode_length': task.max_episode_length, 'actor_only': True,
        'lift_threshold_m': task.touch_lift_target_height, 'lift_hold_steps': task.touch_lift_hold_steps,
        'geometric_margin_m': 0.02, 'seconds': time.monotonic() - started,
        'touch_audit_note': 'Far touches exclude the object bounding sphere; near touches do not prove object collision.',
        'step_note': 'env.reset performs one zero-action simulation step, included in episode statistics.',
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    atomic_json(output / 'result.json', {'complete': True, 'metadata': metadata, 'summary': summarize(rows)})
    print('EVAL_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
