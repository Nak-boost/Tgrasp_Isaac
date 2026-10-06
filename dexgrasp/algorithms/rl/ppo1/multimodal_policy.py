import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


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
        transformer_layers,
        attention_heads,
        feedforward_dim,
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
        token_dim = 64
        if token_dim % attention_heads != 0:
            raise ValueError(
                "Voxel token dimension must be divisible by attention heads"
            )
        self.cnn = nn.Sequential(
            nn.Conv3d(
                channels,
                16,
                kernel_size=4,
                stride=4,
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
            nn.Conv3d(
                32,
                token_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.ELU(),
        )
        with torch.no_grad():
            feature_shape = self.cnn(
                torch.zeros(1, channels, *self.grid_size)
            ).shape[1:]
        self.token_dim = feature_shape[0]
        self.num_spatial_tokens = int(np.prod(feature_shape[1:]))
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, self.token_dim)
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_spatial_tokens + 1,
                self.token_dim,
            )
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(self.token_dim),
        )
        self.projection = nn.Linear(self.token_dim, embedding_dim)
        nn.init.normal_(self.class_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
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
        voxel_features = self.cnn(voxel_map)
        spatial_tokens = voxel_features.flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(
            observations.shape[0], -1, -1
        )
        tokens = torch.cat((class_token, spatial_tokens), dim=1)
        tokens = tokens + self.position_embedding
        encoded_tokens = self.transformer(tokens)
        embedding = self.projection(encoded_tokens[:, 0])
        return self.embedding_normalizer(embedding)


class LocalVoxelObservationEncoder(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, grid_size, channels,
        embedding_normalization,
    ):
        super().__init__()
        expected_dim = channels * int(np.prod(grid_size))
        if input_dim != expected_dim:
            raise ValueError(
                f"local_voxel_map dimension is {input_dim}, expected {expected_dim}"
            )
        self.channels = channels
        self.grid_size = tuple(grid_size)
        self.cnn = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened_dim = self.cnn(
                torch.zeros(1, channels, *self.grid_size)
            ).shape[1]
        self.projection = nn.Linear(flattened_dim, embedding_dim)
        nn.init.orthogonal_(self.projection.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.projection.bias)
        self.embedding_normalizer = build_embedding_normalizer(
            embedding_dim, embedding_normalization,
        )

    def forward(self, observations):
        local_map = observations.reshape(
            observations.shape[0], self.channels, *self.grid_size,
        )
        return self.embedding_normalizer(
            self.projection(self.cnn(local_map))
        )


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
        local_voxel_cfg = tactile_cfg.get("local_voxel", {})
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
                    channels=voxel_cfg.get("channels", 2),
                    transformer_layers=int(
                        voxel_cfg.get("transformer_layers", 2)
                    ),
                    attention_heads=int(
                        voxel_cfg.get("attention_heads", 4)
                    ),
                    feedforward_dim=int(
                        voxel_cfg.get("feedforward_dim", 128)
                    ),
                    embedding_normalization=embedding_normalization,
                )
            elif encoder_type == "local_voxel":
                self.branch_encoders[branch_name] = LocalVoxelObservationEncoder(
                    input_dim=branch_dim,
                    embedding_dim=self.embedding_dim,
                    grid_size=local_voxel_cfg["grid_size"],
                    channels=local_voxel_cfg["channels"],
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

        recurrent_cfg = model_cfg.get("recurrent", {})
        self.is_recurrent = bool(recurrent_cfg.get("enabled", False))
        self.recurrent_hidden_size = int(
            recurrent_cfg.get("hidden_size", 256)
        )
        if self.is_recurrent:
            if self.recurrent_hidden_size < 1:
                raise ValueError("recurrent.hidden_size must be positive")
            self.recurrent_memory = nn.GRUCell(
                input_size=joint_embedding_dim,
                hidden_size=self.recurrent_hidden_size,
            )
            for weight in (
                self.recurrent_memory.weight_ih,
                self.recurrent_memory.weight_hh,
            ):
                for gate_weight in weight.chunk(3, dim=0):
                    nn.init.orthogonal_(gate_weight)
            nn.init.zeros_(self.recurrent_memory.bias_ih)
            nn.init.zeros_(self.recurrent_memory.bias_hh)
            self.recurrent_normalizer = nn.LayerNorm(
                self.recurrent_hidden_size
            )
            policy_input_dim = self.recurrent_hidden_size
        else:
            self.recurrent_memory = None
            self.recurrent_normalizer = nn.Identity()
            policy_input_dim = joint_embedding_dim

        self.actor = self._build_mlp(
            input_dim=policy_input_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=actions_shape[0],
            activation_name=activation_name,
            output_gain=0.01,
        )
        self.critic = self._build_mlp(
            input_dim=policy_input_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation_name=activation_name,
            output_gain=1.0,
        )

        std_cfg = dict(model_cfg.get("state_dependent_std", {}))
        std_cfg.update(tactile_cfg.get("policy_noise", {}))
        self.state_dependent_std = bool(
            std_cfg.get("enabled", False)
        )
        initial_std = float(std_cfg.get("initial", initial_std))
        minimum_std = float(std_cfg.get("minimum", 0.05))
        maximum_std = float(std_cfg.get("maximum", 1.0))
        if not 0.0 < minimum_std < maximum_std:
            raise ValueError(
                "state_dependent_std requires 0 < minimum < maximum"
            )
        if not minimum_std <= initial_std <= maximum_std:
            raise ValueError(
                "init_noise_std must be within the configured std limits"
            )
        self.minimum_log_std = float(np.log(minimum_std))
        self.maximum_log_std = float(np.log(maximum_std))
        self.initial_action_std = initial_std
        self.minimum_action_std = minimum_std
        self.maximum_action_std = maximum_std
        if self.state_dependent_std:
            self.log_std_head = nn.Linear(
                policy_input_dim, actions_shape[0]
            )
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, np.log(initial_std))
            self.register_parameter("log_std", None)
        else:
            self.log_std_head = None
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

    def initial_recurrent_state(self, batch_size, device=None):
        if not self.is_recurrent:
            return None
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            batch_size,
            self.recurrent_hidden_size,
            device=device,
        )

    def apply_recurrent_memory(
        self,
        joint_embedding,
        recurrent_hidden_states,
    ):
        if not self.is_recurrent:
            return joint_embedding, None
        if recurrent_hidden_states is None:
            raise ValueError(
                "A recurrent hidden state is required by this policy"
            )
        next_hidden_states = self.recurrent_memory(
            joint_embedding,
            recurrent_hidden_states,
        )
        return (
            self.recurrent_normalizer(next_hidden_states),
            next_hidden_states,
        )

    def action_parameters(self, policy_features):
        actions_mean = self.actor(policy_features)
        if self.state_dependent_std:
            action_log_std = torch.clamp(
                self.log_std_head(policy_features),
                min=self.minimum_log_std,
                max=self.maximum_log_std,
            )
        else:
            action_log_std = self.log_std.expand_as(actions_mean)
        return actions_mean, action_log_std

    @staticmethod
    def action_distribution(actions_mean, action_log_std):
        return Independent(
            Normal(actions_mean, action_log_std.exp()),
            1,
        )

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations, recurrent_hidden_states=None):
        joint_embedding = self.encode_observations(observations)
        policy_features, next_hidden_states = self.apply_recurrent_memory(
            joint_embedding,
            recurrent_hidden_states,
        )
        actions_mean, action_log_std = self.action_parameters(
            policy_features
        )
        distribution = self.action_distribution(
            actions_mean, action_log_std
        )
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        value = self.critic(policy_features)

        result = (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            action_log_std.detach(),
            observations[:, :self.prop_dim].detach(),
            observations[:, self.prop_dim:].detach(),
        )
        if self.is_recurrent:
            return result + (next_hidden_states.detach(),)
        return result

    @torch.no_grad()
    def act_inference(self, observations, recurrent_hidden_states=None):
        joint_embedding = self.encode_observations(observations)
        policy_features, next_hidden_states = self.apply_recurrent_memory(
            joint_embedding,
            recurrent_hidden_states,
        )
        actions_mean = self.actor(policy_features)
        if self.is_recurrent:
            return actions_mean, next_hidden_states
        return actions_mean

    @torch.no_grad()
    def get_value(self, observations, recurrent_hidden_states=None):
        joint_embedding = self.encode_observations(observations)
        policy_features, _ = self.apply_recurrent_memory(
            joint_embedding,
            recurrent_hidden_states,
        )
        return self.critic(policy_features)

    def evaluate(self, obs_features, state, actions):
        if self.is_recurrent:
            raise RuntimeError(
                "Use evaluate_recurrent for a recurrent policy"
            )
        observations = torch.cat((state, obs_features), dim=1)
        joint_embedding = self.encode_observations(observations)
        actions_mean, action_log_std = self.action_parameters(
            joint_embedding
        )
        distribution = self.action_distribution(
            actions_mean, action_log_std
        )

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = self.critic(joint_embedding)

        return (
            actions_log_prob,
            entropy,
            value,
            actions_mean,
            action_log_std,
        )

    def evaluate_recurrent(
        self,
        obs_features,
        states,
        actions,
        initial_hidden_states,
        dones,
    ):
        if not self.is_recurrent:
            raise RuntimeError(
                "evaluate_recurrent requires a recurrent policy"
            )
        observations = torch.cat((states, obs_features), dim=-1)
        sequence_length, batch_size = observations.shape[:2]
        joint_embeddings = self.encode_observations(
            observations.reshape(sequence_length * batch_size, -1)
        ).reshape(sequence_length, batch_size, -1)

        hidden_states = initial_hidden_states
        recurrent_features = []
        for step in range(sequence_length):
            if step > 0:
                hidden_states = hidden_states * (
                    1.0 - dones[step - 1].float()
                )
            policy_features, hidden_states = (
                self.apply_recurrent_memory(
                    joint_embeddings[step],
                    hidden_states,
                )
            )
            recurrent_features.append(policy_features)

        policy_features = torch.stack(
            recurrent_features, dim=0
        ).reshape(sequence_length * batch_size, -1)
        flat_actions = actions.reshape(
            sequence_length * batch_size, -1
        )
        actions_mean, action_log_std = self.action_parameters(
            policy_features
        )
        distribution = self.action_distribution(
            actions_mean, action_log_std
        )

        return (
            distribution.log_prob(flat_actions),
            distribution.entropy(),
            self.critic(policy_features),
            actions_mean,
            action_log_std,
        )
