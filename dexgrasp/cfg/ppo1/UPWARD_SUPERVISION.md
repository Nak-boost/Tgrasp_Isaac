# Upward Action Supervision

The master branch adds a conditional auxiliary loss to the experiment-B PPO
baseline. The auxiliary loss itself does not change observations, action
execution or reward coefficients. `TGrasp_B` remains the reward-only
comparison. The current master training configuration separately enables a
persistent recurrent policy and state-dependent action noise.

A sample is supervised when the existing tactile predicate reports thumb plus
another finger in contact for three consecutive control steps, the object has
risen less than the configured target height (0.12 m), and no reset is pending.
This predicate does not identify which object caused each tactile contact.

The condition for state `s_t` is stored with `s_t` in the rollout buffer. The
condition returned after stepping to `s_(t+1)` applies to the next action.
During each PPO minibatch update:

```text
L_up = sum(mask * relu(target_action - mu_z(s_t))^2) / max(sum(mask), 1)
L_total = L_PPO + coefficient * L_up
```

Only the Z component is directly supervised. Other outputs can still change
through shared network weights. A zero mask gives zero auxiliary gradient;
means already at or above the target receive no auxiliary penalty. The target
is a normalized hand-base force command, not height or upward velocity.

Configure `learn.upward_action_supervision` in `config.yaml`:

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | Set to `False` for PPO without the auxiliary loss. |
| `action_index` | `2` | Hand-base Z action in the current control layout. |
| `target_action` | `0.3` | Lower bound encouraged for the action mean. |
| `loss_coef` | `0.1` | Initial auxiliary loss coefficient. |
| `trigger_lift_rate` | `0.25` | Completed-episode lift rate that starts decay. |
| `success_window` | `2000` | Number of completed episodes in the rolling window. |
| `decay_iterations` | `2000` | Linear decay duration after the first trigger. |

The full window must be available before decay can start. Without sufficient
lift success, supervision remains active beyond iteration 2000. Once started,
decay proceeds to zero without restarting when success drops.

Existing training commands, including `--history_length 4`, `8`, or `16`, work
unchanged. Use a new log directory when comparing this variant to experiment-B.
Checkpoints created before the recurrent policy was enabled are not directly
loadable by the current network structure. New checkpoints also save
`model_N.pt.supervision.json`; keep it beside the weights when resuming to
preserve the decay state. Loading weights without that file starts a fresh
supervision schedule. The original PPO optimizer-resume behavior is unchanged.

TensorBoard records `Loss/upward_action_supervision` and `Aux/` metrics for the
coefficient, eligible-sample fraction, eligible rollout mean Z action, rolling
lift rate and whether decay has started. The rolling lift rate is in [0, 1].
An empty eligible set reports a mean Z of zero; inspect the fraction as well.
