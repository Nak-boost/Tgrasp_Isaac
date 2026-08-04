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
    normalization,
):
    normalization = str(normalization).lower()
    if normalization in {"none", "identity"}:
        return nn.Identity()
    if normalization == "layer_norm":
        return nn.LayerNorm(embedding_dim)
    raise ValueError(
        f"Unsupported embedding normalization: {normalization}"
    )


class IdentityObservationEncoder(nn.Module):
    def forward(self, observations):
        return observations


class ProprioceptionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dims,
        activation_name,
        embedding_normalization,
    ):
        super().__init__()
        self.encoder = build_mlp(
            input_dim,
            hidden_dims,
            embedding_dim,
            activation_name,
        )
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_normalization,
        )

    def forward(self, observations):
        embedding = self.encoder(observations)
        return self.embedding_normalizer(embedding)


class TouchObservationEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dims,
        activation_name,
        embedding_normalization,
    ):
        super().__init__()

        # Raw 0/1 touch values enter the MLP directly. LayerNorm is
        # applied only after the touch pattern has been encoded.
        self.encoder = build_mlp(
            input_dim,
            hidden_dims,
            embedding_dim,
            activation_name,
        )
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_normalization,
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
        embedding_normalization,
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
            embedding_normalization,
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
        embedding_normalization,
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
            embedding_normalization,
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
        embedding_normalization,
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
            nn.Conv3d(
                channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, 16),
            nn.ELU(),
            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ELU(),
        )
        with torch.no_grad():
            feature_shape = self.cnn(
                torch.zeros(1, channels, *self.grid_size)
            ).shape[1:]
        self.projection = nn.Linear(
            int(np.prod(feature_shape)),
            embedding_dim,
        )
        nn.init.orthogonal_(self.projection.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.projection.bias)
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_normalization,
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


class PointNetGlobalFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dims,
        activation_name,
        embedding_normalization,
    ):
        super().__init__()

        self.input_normalizer = nn.LayerNorm(input_dim)
        self.encoder = build_mlp(
            input_dim,
            hidden_dims,
            embedding_dim,
            activation_name,
        )
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_normalization,
        )

    def forward(self, observations):
        normalized_features = self.input_normalizer(observations)
        embedding = self.encoder(normalized_features)
        return self.embedding_normalizer(embedding)


class DexRepInteractionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        distance_dim,
        occupancy_grid_size,
        keypoint_count,
        local_feature_dim,
        embedding_normalization,
    ):
        super().__init__()

        self.distance_dim = distance_dim
        self.occupancy_grid_size = tuple(occupancy_grid_size)
        self.occupancy_dim = int(
            np.prod(self.occupancy_grid_size)
        )
        self.sensor_dim = self.distance_dim + self.occupancy_dim
        self.keypoint_count = keypoint_count
        self.local_feature_dim = local_feature_dim
        self.local_dim = (
            self.keypoint_count * self.local_feature_dim
        )

        expected_dim = self.sensor_dim + self.local_dim
        if input_dim != expected_dim:
            raise ValueError(
                f"DexRep interaction dimension is {input_dim}, "
                f"expected {self.sensor_dim} + {self.local_dim}"
            )

        self.distance_encoder = build_mlp(
            self.distance_dim,
            [128],
            64,
            "elu",
        )
        self.occupancy_encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.GroupNorm(2, 8),
            nn.ELU(),
            nn.Conv3d(
                8,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, 16),
            nn.ELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.local_point_encoder = build_mlp(
            self.local_feature_dim,
            [128],
            128,
            "elu",
        )
        self.local_projection = build_mlp(
            256,
            [128],
            128,
            "elu",
        )
        self.fusion = build_mlp(
            224,
            [256],
            embedding_dim,
            "elu",
        )
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim,
            embedding_normalization,
        )

    def forward(self, observations):
        distance = observations[:, :self.distance_dim]
        occupancy_end = self.sensor_dim
        occupancy = observations[
            :, self.distance_dim:occupancy_end
        ].reshape(
            observations.shape[0],
            1,
            *self.occupancy_grid_size,
        )
        local_features = observations[
            :, occupancy_end:
        ].reshape(
            observations.shape[0],
            self.keypoint_count,
            self.local_feature_dim,
        )

        distance_embedding = self.distance_encoder(distance)
        occupancy_embedding = self.occupancy_encoder(
            occupancy
        ).flatten(1)

        encoded_local_points = self.local_point_encoder(
            local_features.reshape(
                -1, self.local_feature_dim
            )
        ).reshape(
            observations.shape[0],
            self.keypoint_count,
            -1,
        )
        local_max = encoded_local_points.max(dim=1).values
        local_mean = encoded_local_points.mean(dim=1)
        local_embedding = self.local_projection(
            torch.cat((local_max, local_mean), dim=1)
        )

        embedding = self.fusion(
            torch.cat(
                (
                    distance_embedding,
                    occupancy_embedding,
                    local_embedding,
                ),
                dim=1,
            )
        )
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

        encoder_specs = env_cfg.get("obs_encoders", {})
        tactile_cfg = env_cfg.get("tactile", {})
        voxel_cfg = env_cfg.get("tactile", {}).get("voxel", {})
        history_cfg = tactile_cfg.get("history", {})
        geometry_cfg = tactile_cfg.get(
            "oracle_geometry", {}
        )

        self.branch_encoders = nn.ModuleDict()
        self.branch_output_dims = {}
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
            embedding_normalization = encoder_spec.get(
                "embedding_normalization", "none"
            )

            if encoder_type == "voxel":
                self.branch_encoders[branch_name] = VoxelObservationEncoder(
                    input_dim=branch_dim,
                    embedding_dim=self.embedding_dim,
                    grid_size=voxel_cfg["grid_size"],
                    channels=voxel_cfg.get("channels", 4),
                    embedding_normalization=embedding_normalization,
                )
            elif encoder_type == "pointnet_global":
                self.branch_encoders[branch_name] = (
                    PointNetGlobalFeatureEncoder(
                        input_dim=branch_dim,
                        embedding_dim=self.embedding_dim,
                        hidden_dims=encoder_spec.get(
                            "hidden_dims", [512, 256]
                        ),
                        activation_name=encoder_spec.get(
                            "activation", "elu"
                        ),
                        embedding_normalization=embedding_normalization,
                    )
                )
            elif encoder_type == "dexrep_interaction":
                self.branch_encoders[branch_name] = (
                    DexRepInteractionEncoder(
                        input_dim=branch_dim,
                        embedding_dim=self.embedding_dim,
                        distance_dim=geometry_cfg.get(
                            "dexrep_distance_dim", 80
                        ),
                        occupancy_grid_size=geometry_cfg.get(
                            "dexrep_occupancy_grid_size",
                            [10, 10, 10],
                        ),
                        keypoint_count=geometry_cfg.get(
                            "dexrep_keypoint_count", 20
                        ),
                        local_feature_dim=geometry_cfg.get(
                            "dexrep_local_feature_dim", 64
                        ),
                        embedding_normalization=embedding_normalization,
                    )
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
                        embedding_normalization=embedding_normalization,
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
                        embedding_normalization=embedding_normalization,
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
                        embedding_normalization=embedding_normalization,
                    )
                )
            elif encoder_type == "identity":
                self.branch_encoders[branch_name] = (
                    IdentityObservationEncoder()
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
                        embedding_normalization=(
                            embedding_normalization
                        ),
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported encoder '{encoder_type}' "
                    f"for branch '{branch_name}'"
                )

            self.branch_output_dims[branch_name] = (
                branch_dim
                if encoder_type == "identity"
                else self.embedding_dim
            )

        actor_hidden_dims = model_cfg.get(
            "pi_hid_sizes", [256, 256, 256]
        )
        critic_hidden_dims = model_cfg.get(
            "vf_hid_sizes", [256, 256, 256]
        )
        activation_name = model_cfg.get("activation", "elu")
        joint_embedding_dim = sum(self.branch_output_dims.values())

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


    def action_distribution(self, actions_mean):
        action_std = self.log_std.exp()
        scale_tril = torch.diag(action_std)
        return MultivariateNormal(actions_mean, scale_tril=scale_tril)

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
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
        joint_embedding = self.encode_observations(observations)
        return self.actor(joint_embedding)

    def evaluate(self, obs_features, state, actions):
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
