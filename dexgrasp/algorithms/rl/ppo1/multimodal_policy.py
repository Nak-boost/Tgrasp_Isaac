import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


def get_activation(activation_name):
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if activation_name not in activations:
        raise ValueError(f"Unsupported activation: {activation_name}")
    return activations[activation_name]()


def initialize_linear_layers(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)


def build_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    activation_name="elu",
):
    layers = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous_dim, hidden_dim),
                get_activation(activation_name),
            ]
        )
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, output_dim))
    network = nn.Sequential(*layers)
    initialize_linear_layers(network)
    return network


def build_embedding_normalizer(
    embedding_dim,
    enabled,
    momentum,
):
    if not enabled:
        return nn.Identity()
    return nn.BatchNorm1d(
        embedding_dim,
        momentum=momentum,
        affine=True,
        track_running_stats=True,
    )


class ProprioceptionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dims,
        activation_name,
    ):
        super().__init__()
        self.encoder = build_mlp(
            input_dim,
            hidden_dims,
            embedding_dim,
            activation_name,
        )

    def forward(self, observations):
        return self.encoder(observations)


class TouchObservationEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dims,
        activation_name,
        embedding_batch_norm,
        batch_norm_momentum,
    ):
        super().__init__()

        # Raw 0/1 touch values enter the MLP directly. BatchNorm is only
        # applied after the touch pattern has been encoded.
        self.encoder = build_mlp(
            input_dim,
            hidden_dims,
            embedding_dim,
            activation_name,
        )
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_batch_norm,
            batch_norm_momentum,
        )

    def forward(self, observations):
        embedding = self.encoder(observations)
        return self.embedding_normalizer(embedding)


class ObjectObservationEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dims,
        activation_name,
        embedding_batch_norm,
        batch_norm_momentum,
    ):
        super().__init__()

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    get_activation(activation_name),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)
        initialize_linear_layers(self.encoder)
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_batch_norm,
            batch_norm_momentum,
        )

    def forward(self, observations):
        embedding = self.encoder(observations)
        return self.embedding_normalizer(embedding)


class HistoryObservationEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        history_length,
        frame_dim,
        frame_hidden_dims,
        frame_embedding_dim,
        recurrent_hidden_dim,
        activation_name,
        embedding_batch_norm,
        batch_norm_momentum,
    ):
        super().__init__()

        expected_dim = history_length * frame_dim
        if input_dim != expected_dim:
            raise ValueError(
                f"touch_history dimension is {input_dim}, expected "
                f"{history_length} * {frame_dim} = {expected_dim}"
            )

        self.history_length = history_length
        self.frame_dim = frame_dim
        self.frame_encoder = build_mlp(
            frame_dim,
            frame_hidden_dims,
            frame_embedding_dim,
            activation_name,
        )
        self.temporal_encoder = nn.GRU(
            input_size=frame_embedding_dim,
            hidden_size=recurrent_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.projection = nn.Linear(
            recurrent_hidden_dim,
            embedding_dim,
        )
        nn.init.orthogonal_(self.projection.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.projection.bias)
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_batch_norm,
            batch_norm_momentum,
        )

    def forward(self, observations):
        history = observations.reshape(
            observations.shape[0],
            self.history_length,
            self.frame_dim,
        )
        frame_embeddings = self.frame_encoder(
            history.reshape(-1, self.frame_dim)
        ).reshape(
            observations.shape[0],
            self.history_length,
            -1,
        )
        _, final_hidden = self.temporal_encoder(frame_embeddings)
        embedding = self.projection(final_hidden[-1])
        return self.embedding_normalizer(embedding)


class VoxelObservationEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        grid_size,
        channels,
        embedding_batch_norm,
        batch_norm_momentum,
    ):
        super().__init__()

        expected_dim = channels * int(np.prod(grid_size))
        if input_dim != expected_dim:
            raise ValueError(
                f"voxel_map dimension is {input_dim}, expected "
                f"{channels} * {list(grid_size)} = {expected_dim}"
            )

        self.channels = channels
        self.grid_size = tuple(grid_size)
        self.cnn = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projection = nn.Linear(32, embedding_dim)
        nn.init.orthogonal_(self.projection.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.projection.bias)
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_batch_norm,
            batch_norm_momentum,
        )

    def forward(self, observations):
        voxel_map = observations.reshape(
            observations.shape[0],
            self.channels,
            *self.grid_size,
        )
        voxel_features = self.cnn(voxel_map).flatten(1)
        embedding = self.projection(voxel_features)
        return self.embedding_normalizer(embedding)


class ActorCriticMultimodal(nn.Module):
    def __init__(
        self,
        obs_shape,
        actions_shape,
        initial_std,
        model_cfg,
        encoder_cfg,
        env_cfg,
    ):
        super().__init__()

        self.obs_dims = dict(env_cfg["obs_dim"])
        self.obs_names = list(self.obs_dims)

        if not self.obs_names or self.obs_names[0] != "prop":
            raise ValueError("obs_dim must contain 'prop' as its first branch")

        configured_obs_dim = sum(self.obs_dims.values())
        if configured_obs_dim != obs_shape[0]:
            raise ValueError(
                f"Observation branches sum to {configured_obs_dim}, "
                f"but environment observation dimension is {obs_shape[0]}"
            )

        self.prop_dim = self.obs_dims["prop"]
        self.embedding_dim = encoder_cfg.get("emb_dim", 128)
        self.embedding_bn_momentum = encoder_cfg.get(
            "embedding_bn_momentum", 0.01
        )

        encoder_specs = env_cfg.get("obs_encoders", {})
        tactile_cfg = env_cfg.get("tactile", {})
        voxel_cfg = env_cfg.get("tactile", {}).get("voxel", {})
        history_cfg = tactile_cfg.get("history", {})

        self.branch_encoders = nn.ModuleDict()
        for branch_name, branch_dim in self.obs_dims.items():
            encoder_spec = encoder_specs.get(
                branch_name,
                {
                    "type": (
                        "voxel"
                        if branch_name == "voxel_map"
                        else "mlp"
                    )
                },
            )

            if isinstance(encoder_spec, str):
                encoder_spec = {"type": encoder_spec}

            encoder_type = encoder_spec.get("type", "mlp")
            embedding_batch_norm = encoder_spec.get(
                "embedding_batch_norm", False
            )

            if encoder_type == "voxel":
                self.branch_encoders[branch_name] = VoxelObservationEncoder(
                    input_dim=branch_dim,
                    embedding_dim=self.embedding_dim,
                    grid_size=voxel_cfg["grid_size"],
                    channels=voxel_cfg.get("channels", 4),
                    embedding_batch_norm=embedding_batch_norm,
                    batch_norm_momentum=self.embedding_bn_momentum,
                )
            elif encoder_type == "touch":
                self.branch_encoders[branch_name] = (
                    TouchObservationEncoder(
                        input_dim=branch_dim,
                        embedding_dim=self.embedding_dim,
                        hidden_dims=encoder_spec.get(
                            "hidden_dims", [32, 64]
                        ),
                        activation_name=encoder_spec.get(
                            "activation", "elu"
                        ),
                        embedding_batch_norm=embedding_batch_norm,
                        batch_norm_momentum=self.embedding_bn_momentum,
                    )
                )
            elif encoder_type == "object":
                self.branch_encoders[branch_name] = (
                    ObjectObservationEncoder(
                        input_dim=branch_dim,
                        embedding_dim=self.embedding_dim,
                        hidden_dims=encoder_spec.get(
                            "hidden_dims", [128, 128]
                        ),
                        activation_name=encoder_spec.get(
                            "activation", "elu"
                        ),
                        embedding_batch_norm=embedding_batch_norm,
                        batch_norm_momentum=self.embedding_bn_momentum,
                    )
                )
            elif encoder_type == "history":
                self.branch_encoders[branch_name] = (
                    HistoryObservationEncoder(
                        input_dim=branch_dim,
                        embedding_dim=self.embedding_dim,
                        history_length=history_cfg["length"],
                        frame_dim=history_cfg["frame_dim"],
                        frame_hidden_dims=encoder_spec.get(
                            "frame_hidden_dims", [128]
                        ),
                        frame_embedding_dim=encoder_spec.get(
                            "frame_embedding_dim", 128
                        ),
                        recurrent_hidden_dim=encoder_spec.get(
                            "recurrent_hidden_dim", 128
                        ),
                        activation_name=encoder_spec.get(
                            "activation", "elu"
                        ),
                        embedding_batch_norm=embedding_batch_norm,
                        batch_norm_momentum=self.embedding_bn_momentum,
                    )
                )
            elif encoder_type == "mlp":
                self.branch_encoders[branch_name] = (
                    ProprioceptionEncoder(
                        input_dim=branch_dim,
                        embedding_dim=self.embedding_dim,
                        hidden_dims=encoder_spec.get(
                            "hidden_dims", [256, 128]
                        ),
                        activation_name=encoder_spec.get(
                            "activation", "elu"
                        ),
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported encoder '{encoder_type}' "
                    f"for branch '{branch_name}'"
                )

        actor_hidden_dims = model_cfg.get(
            "pi_hid_sizes", [256, 256, 256]
        )
        critic_hidden_dims = model_cfg.get(
            "vf_hid_sizes", [256, 256, 256]
        )
        activation_name = model_cfg.get("activation", "elu")
        joint_embedding_dim = len(self.obs_dims) * self.embedding_dim

        self.actor = self._build_mlp(
            input_dim=joint_embedding_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=actions_shape[0],
            activation_name=activation_name,
            output_gain=0.01,
        )
        self.critic = self._build_mlp(
            input_dim=joint_embedding_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation_name=activation_name,
            output_gain=1.0,
        )

        self.log_std = nn.Parameter(
            np.log(initial_std) * torch.ones(*actions_shape)
        )

    @staticmethod
    def _build_mlp(
        input_dim,
        hidden_dims,
        output_dim,
        activation_name,
        output_gain,
    ):
        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            linear = nn.Linear(previous_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(linear.bias)
            layers.extend([linear, get_activation(activation_name)])
            previous_dim = hidden_dim

        output = nn.Linear(previous_dim, output_dim)
        nn.init.orthogonal_(output.weight, gain=output_gain)
        nn.init.zeros_(output.bias)
        layers.append(output)
        return nn.Sequential(*layers)

    def split_observations(self, observations):
        branch_observations = {}
        start = 0

        for branch_name, branch_dim in self.obs_dims.items():
            end = start + branch_dim
            branch_observations[branch_name] = observations[:, start:end]
            start = end

        if start != observations.shape[1]:
            raise ValueError(
                f"Consumed {start} observation values, "
                f"received {observations.shape[1]}"
            )

        return branch_observations

    def encode_observations(self, observations):
        branch_observations = self.split_observations(observations)
        embeddings = []

        for branch_name in self.obs_names:
            embedding = self.branch_encoders[branch_name](
                branch_observations[branch_name]
            )
            embeddings.append(embedding)

        return torch.cat(embeddings, dim=1)

    def set_normalizers_training(self, training):
        for module in self.branch_encoders.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train(training)

    def action_distribution(self, actions_mean):
        action_std = self.log_std.exp()
        scale_tril = torch.diag(action_std)
        return MultivariateNormal(actions_mean, scale_tril=scale_tril)

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.set_normalizers_training(False)
        joint_embedding = self.encode_observations(observations)
        actions_mean = self.actor(joint_embedding)
        distribution = self.action_distribution(actions_mean)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        value = self.critic(joint_embedding)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
            observations[:, :self.prop_dim].detach(),
            observations[:, self.prop_dim:].detach(),
        )

    @torch.no_grad()
    def act_inference(self, observations):
        self.set_normalizers_training(False)
        joint_embedding = self.encode_observations(observations)
        return self.actor(joint_embedding)

    def evaluate(self, obs_features, state, actions):
        self.set_normalizers_training(True)
        observations = torch.cat((state, obs_features), dim=1)
        joint_embedding = self.encode_observations(observations)
        actions_mean = self.actor(joint_embedding)
        distribution = self.action_distribution(actions_mean)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = self.critic(joint_embedding)

        return (
            actions_log_prob,
            entropy,
            value,
            actions_mean,
            self.log_std.repeat(actions_mean.shape[0], 1),
        )
