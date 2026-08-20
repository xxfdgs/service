"""
OneHotEmbedGPSModel: Dual-pathway polymer property predictor.

Components 1-4 (IL, HL, Chol, PEG) — low-diversity, near-fixed molecules:
    one-hot encode by molecule identity → learnable dense embeddings.

Component 5 (Fifth) — high-diversity, variable molecules:
    full GraphGPS encoder (GINE+Transformer, no augmented descriptors).

Fusion: concatenate 5 embeddings + 5 ratios → MLP → multi-property predictions.
"""

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (
    new_layer_config,
    BatchNorm1dNode,
)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
from graphgps.component_vocab import (
    get_vocab_id,
    build_component_embeddings,
)
from graphgps.component_aux import component_aux_enabled


# ---------------------------------------------------------------------------
# Component-5 GraphGPS encoder (reuses FeatureEncoder + GPSLayers from GPSModel)
# ---------------------------------------------------------------------------

class Comp5GraphEncoder(nn.Module):
    """Graph encoder for the fifth component — same as GPSModel minus the head."""

    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

        # Node encoder
        if cfg.dataset.node_encoder:
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1,
                                     has_act=False, has_bias=False, cfg=cfg))
            self.dim_in = cfg.gnn.dim_inner

        # Edge encoder
        if cfg.dataset.edge_encoder:
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1,
                                     has_act=False, has_bias=False, cfg=cfg))

        # Pre-MP layers
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == self.dim_in, \
            "The inner and hidden dims must match."

        # GPS layers
        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except Exception:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


# ---------------------------------------------------------------------------
# Main dual-pathway model
# ---------------------------------------------------------------------------

@register_network('OneHotEmbedGPS')
class OneHotEmbedGPSModel(nn.Module):
    """
    Dual-pathway model for 5-component polymer property prediction.

    Pathway A (components 1-4):
        Identify molecule by atom count → one-hot embedding → ratio modulation.

    Pathway B (component 5):
        Full GraphGPS encoder → graph pooling → graph embedding.

    Fusion:
        Concat(emb1..emb5, ratios) → 3-layer MLP → predictions.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.hidden_dim = cfg.gt.dim_hidden
        self.num_components = 5
        self.out_dim = cfg.property_num
        self.output_activation = str(getattr(cfg.model, 'output_activation', 'identity'))
        if self.output_activation not in {'identity', 'sigmoid'}:
            raise ValueError(
                'model.output_activation must be either "identity" or "sigmoid".')

        # --- Pathway A: one-hot embeddings for components 1-4 ---
        self.comp_embeddings = build_component_embeddings(
            self.hidden_dim, cfg.component_vocab_sizes)
        if cfg.use_fifth_identity_embedding:
            self.fifth_component_embedding = nn.Embedding(
                int(cfg.fifth_component_vocab_size), self.hidden_dim)
        if cfg.use_fifth_class_embedding:
            self.fifth_class_embedding = nn.Embedding(
                int(cfg.fifth_class_vocab_size), self.hidden_dim)

        # Optional input-only molecular features (Morgan bits plus bounded
        # RDKit descriptors).  They complement the exact identity embeddings
        # of the low-diversity components and the GraphGPS embedding of the
        # fifth component without changing the historical O1 path when the
        # feature flag is off.
        if cfg.use_component_aux_features:
            self.aux_feature_encoder = nn.Sequential(
                nn.Linear(cfg.component_aux_dim, self.hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            )
        self.fifth_semantic_feature_dim = (
            int(cfg.fifth_semantic_feature_dim)
            if cfg.use_fifth_semantic_features else 0)
        if self.fifth_semantic_feature_dim:
            self.fifth_semantic_encoder = nn.Sequential(
                nn.Linear(self.fifth_semantic_feature_dim, self.hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            )
        self.use_fifth_structured_features = bool(cfg.use_fifth_structured_features)
        if self.use_fifth_structured_features:
            self.fifth_aa_embedding = nn.Embedding(int(cfg.fifth_aa_vocab_size), int(cfg.fifth_aa_embedding_dim))
            self.fifth_terminal_embedding = nn.Embedding(int(cfg.fifth_terminal_vocab_size), int(cfg.fifth_terminal_embedding_dim))
            structured_dim = 2 * int(cfg.fifth_aa_embedding_dim) + int(cfg.fifth_terminal_embedding_dim) + 2
            self.fifth_structured_encoder = nn.Sequential(nn.Linear(structured_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))

        # Ratio-aware modulation for each component 1-4:
        # ratio_feat = [r, sqrt(r), log1p(50r), 1(r>0)] → gamma, beta
        self.ratio_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            )
            for _ in range(4)  # only for comps 1-4
        ])
        # Component 5 is graph-encoded, so it needs its own modulation block
        # rather than sharing a categorical component block.  This is an
        # opt-in ablation: historical models retain the raw pooled embedding.
        if cfg.use_fifth_ratio_modulation:
            self.fifth_ratio_modulator = nn.Sequential(
                nn.Linear(4, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            )

        # --- Pathway B: GraphGPS encoder for component 5 ---
        self.comp5_encoder = Comp5GraphEncoder(dim_in)
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        # Stage-8 adds a second Fifth encoder that carries only the frozen
        # Stage-4 PT-DF structural representation.  The historical
        # ``comp5_encoder`` remains independently initialized and trainable.
        self.frozen_comp5_aux_enable = bool(
            getattr(cfg.model, 'frozen_comp5_aux_enable', False)
        )
        self.frozen_comp5_aux_dim = (
            self.hidden_dim if self.frozen_comp5_aux_enable else 0
        )
        if self.frozen_comp5_aux_enable:
            self.frozen_comp5_aux_encoder = Comp5GraphEncoder(dim_in)
            for parameter in self.frozen_comp5_aux_encoder.parameters():
                parameter.requires_grad_(False)
            self.frozen_comp5_aux_encoder.eval()

        # --- Fusion MLP ---
        # 5 embeddings + 5 ratios = 5*hidden_dim + 5
        self.mordred_feature_dim = cfg.mordred_feature_dim if cfg.use_mordred_features else 0
        self.fifth_mechanistic_descriptor_dim = (
            cfg.fifth_mechanistic_descriptor_dim
            if cfg.use_fifth_mechanistic_descriptors else 0
        )
        self.fifth_only_fusion = bool(
            getattr(cfg.model, 'fifth_only_fusion', False))
        mordred_multiplier = 1 if cfg.mordred_fifth_only else self.num_components
        self.ratio_polynomial_features = bool(
            getattr(cfg.model, 'ratio_polynomial_features', False))
        # Besides raw composition fractions, the optional compact basis gives
        # the fusion head direct access to curvature and pair interactions.
        # It is especially useful with small formulation data, where learning
        # every second-order interaction through a wide MLP is inefficient.
        self.ratio_basis_dim = 20 if self.ratio_polynomial_features else 0
        if self.fifth_only_fusion:
            fusion_in_dim = (self.hidden_dim + 1 + self.mordred_feature_dim
                             + self.fifth_mechanistic_descriptor_dim)
        else:
            fusion_in_dim = (
                self.num_components * self.hidden_dim + self.num_components
                + self.ratio_basis_dim
                + mordred_multiplier * self.mordred_feature_dim
                + self.fifth_mechanistic_descriptor_dim)
        # Keep the complete existing fusion layout and append exactly one
        # frozen Fifth structural vector when Stage-8 is enabled.
        fusion_in_dim += self.frozen_comp5_aux_dim
        dropout = float(getattr(cfg.gt, 'dropout', 0.0))
        act_fn = nn.SiLU

        # ``gated_concat`` retains every component representation but learns
        # an input-dependent relevance weight for each component before the
        # regular fusion MLP. ``residual_concat`` retains the plain MLP and
        # adds a direct linear readout from the complete fusion input, so the
        # nonlinear stack can focus on molecular interaction residuals.
        # Other fusion types preserve the historical plain-concatenation path.
        self.fusion_type = str(getattr(cfg.model, 'fusion_type', 'concat_mlp'))
        if (
            self.fifth_only_fusion
            and self.fusion_type in {'gated_concat', 'attention_concat'}
        ):
            raise ValueError(
                'fifth_only_fusion requires a non-token fusion type.')
        if self.fusion_type == 'gated_concat':
            self.component_gate = nn.Sequential(
                nn.Linear(fusion_in_dim, self.hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.num_components),
                nn.Sigmoid(),
            )
        elif self.fusion_type == 'attention_concat':
            # Five component tokens exchange information before concatenation;
            # learned positions preserve each component's semantic role.
            self.component_position = nn.Parameter(
                torch.zeros(1, self.num_components, self.hidden_dim))
            self.component_attention = nn.MultiheadAttention(
                self.hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
            self.component_attention_norm = nn.LayerNorm(self.hidden_dim)

        # Keep the historical 256 → 128 → 64 head unless an experiment
        # explicitly requests a different fusion width.  This makes the
        # runner's --fusion-hidden-dim control meaningful for OneHotEmbedGPS
        # without altering any existing checkpoints/configurations.
        fusion_hidden_dim = getattr(cfg.model, 'fusion_hidden_dim', None)
        fusion_hidden_dim = 256 if fusion_hidden_dim is None else int(fusion_hidden_dim)
        if fusion_hidden_dim < 4:
            raise ValueError('model.fusion_hidden_dim must be at least 4.')
        fusion_mid_dim = max(2, fusion_hidden_dim // 2)
        fusion_last_dim = max(1, fusion_hidden_dim // 4)

        fusion_layers = [
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_mid_dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(fusion_mid_dim, fusion_last_dim),
            act_fn(),
        ]
        # O14-A attaches a deliberately minimal binary high-Norm logit to the
        # exact final shared fusion representation used by its regression
        # readout.  This branch is strictly opt-in so legacy model state-dict
        # names and two-value forward returns remain unchanged.
        self.use_norm_threshold_head = bool(getattr(cfg, 'use_norm_threshold_head', False))
        if self.use_norm_threshold_head:
            if self.out_dim != 1 or cfg.model.target_specific_heads:
                raise ValueError(
                    'The O14-A Norm threshold head requires one non-target-specific regression output.')
            self.fusion_backbone = nn.Sequential(*fusion_layers)
            self.regression_head = nn.Linear(fusion_last_dim, self.out_dim)
            self.norm_threshold_head = nn.Linear(fusion_last_dim, 1)
            if self.fusion_type == 'residual_concat':
                self.fusion_linear_skip = nn.Linear(fusion_in_dim, self.out_dim)
                nn.init.zeros_(self.fusion_linear_skip.weight)
                nn.init.zeros_(self.fusion_linear_skip.bias)
        elif cfg.model.target_specific_heads:
            # A genuinely separate small head for each property.  The
            # historical multi-output linear layer remains the default path.
            self.fusion_backbone = nn.Sequential(*fusion_layers)
            target_hidden_dim = max(1, fusion_last_dim // 2)
            self.target_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(fusion_last_dim, target_hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                    nn.Linear(target_hidden_dim, 1),
                )
                for _ in range(self.out_dim)
            ])
        else:
            self.fusion = nn.Sequential(
                *fusion_layers,
                nn.Linear(fusion_last_dim, self.out_dim),
            )
            if self.fusion_type == 'residual_concat':
                self.fusion_linear_skip = nn.Linear(fusion_in_dim, self.out_dim)
                # Start exactly from the ordinary fusion MLP.  The residual
                # readout then earns a non-zero contribution through training
                # instead of perturbing an already well-behaved baseline at
                # initialization.
                nn.init.zeros_(self.fusion_linear_skip.weight)
                nn.init.zeros_(self.fusion_linear_skip.bias)

    def train(self, mode=True):
        """Keep the Stage-8 auxiliary encoder in inference mode.

        ``nn.Module.train(True)`` recursively flips child BatchNorm/dropout
        modules.  The frozen branch must stay exactly in Stage-4 eval mode.
        """
        super().train(mode)
        if self.frozen_comp5_aux_enable:
            self.frozen_comp5_aux_encoder.eval()
        return self

    @staticmethod
    def _build_ratio_feature(ratio):
        """ratio: [B, 1] → ratio_feat: [B, 4]"""
        eps = 1e-8
        return torch.cat([
            ratio,
            torch.sqrt(ratio.clamp_min(0.0) + eps),
            torch.log1p(50.0 * ratio.clamp_min(0.0)),
            (ratio > 0).float(),
        ], dim=1)

    @staticmethod
    def _build_composition_basis(ratios):
        """Return powers and all pairwise interactions of five fractions."""
        squares = ratios.square()
        pairs = [ratios[:, left:left + 1] * ratios[:, right:right + 1]
                 for left in range(ratios.size(1))
                 for right in range(left + 1, ratios.size(1))]
        return torch.cat([squares, *pairs, torch.sqrt(ratios.clamp_min(0.0))], dim=1)

    def _encode_components_1_to_4(self, batch, comp_idx):
        """Encode one of the first 4 components via one-hot embedding + ratio modulation.

        Args:
            batch: PyG DataBatch for this component.
            comp_idx: 0-based component index (0=IL, 1=HL, 2=Chol, 3=PEG).

        Returns:
            embedding: [B, hidden_dim]
        """
        B = batch.num_graphs

        # New datasets carry input-derived canonical-SMILES IDs.  Retain the
        # atom-count fallback solely for old processed caches.
        if hasattr(batch, 'component_vocab_id'):
            vocab_ids = batch.component_vocab_id.view(-1).long()
        else:
            num_nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
            vocab_ids = get_vocab_id(comp_idx + 1, num_nodes_per_graph)
            vocab_ids[vocab_ids < 0] = 0

        # Fail with an explicit diagnostic instead of relying on the less
        # informative device-side nn.Embedding index error.
        embedding = self.comp_embeddings[f'comp{comp_idx + 1}']
        invalid = (vocab_ids < 0) | (vocab_ids >= embedding.num_embeddings)
        if invalid.any():
            bad_ids = sorted(set(vocab_ids[invalid].detach().cpu().tolist()))
            raise IndexError(
                f'Component {comp_idx + 1} vocabulary IDs {bad_ids} exceed '
                f'embedding size {embedding.num_embeddings}.')

        # One-hot embedding lookup
        emb = embedding(vocab_ids)  # [B, hidden_dim]

        # Ratio modulation
        ratio = batch.ratio.view(B, 1).float()
        ratio_feat = self._build_ratio_feature(ratio)                # [B, 4]
        gamma, beta = self.ratio_modulators[comp_idx](ratio_feat).chunk(2, dim=-1)
        emb = emb * (1.0 + torch.tanh(gamma)) + beta

        if component_aux_enabled(cfg, comp_idx):
            aux_feat = batch.aux_feat.view(B, -1).float()
            if aux_feat.size(1) != cfg.component_aux_dim:
                raise ValueError(
                    f"Expected {cfg.component_aux_dim} auxiliary features, "
                    f"got {aux_feat.size(1)}.")
            emb = emb + self.aux_feature_encoder(aux_feat)

        return emb, ratio

    def _apply_index(self, batch):
        pred = batch.graph_feature
        out_dim = self.out_dim

        if out_dim == 1:
            label = batch.y.view(-1).unsqueeze(1)
        elif out_dim == 2:
            y0 = getattr(batch, 'y').view(-1)
            y1 = getattr(batch, 'y1').view(-1)
            label = torch.stack([y0, y1], dim=1)
        elif out_dim == 4:
            y0 = getattr(batch, 'y').view(-1)
            y1 = getattr(batch, 'y1').view(-1)
            y2 = getattr(batch, 'y2').view(-1)
            y3 = getattr(batch, 'y3').view(-1)
            label = torch.stack([y0, y1, y2, y3], dim=1)
        elif out_dim == 6:
            y0 = getattr(batch, 'y').view(-1)
            y1 = getattr(batch, 'y1').view(-1)
            y2 = getattr(batch, 'y2').view(-1)
            y3 = getattr(batch, 'y3').view(-1)
            y4 = getattr(batch, 'y4').view(-1)
            y5 = getattr(batch, 'y5').view(-1)
            label = torch.stack([y0, y1, y2, y3, y4, y5], dim=1)
        else:
            label = batch.y.view(-1, 1)

        return pred, label

    def forward(self, data1, data2, data3, data4, data5):
        # --- Pathway A: one-hot embed components 1-4 ---
        emb1, ratio1 = self._encode_components_1_to_4(data1, 0)
        emb2, ratio2 = self._encode_components_1_to_4(data2, 1)
        emb3, ratio3 = self._encode_components_1_to_4(data3, 2)
        emb4, ratio4 = self._encode_components_1_to_4(data4, 3)

        # --- Pathway B: GraphGPS for component 5 ---
        emb5_frozen = None
        if self.frozen_comp5_aux_enable:
            # Comp5GraphEncoder mutates a PyG Batch, so consume a clone before
            # the task-specific encoder processes the original Fifth graph.
            frozen_input = data5.clone()
            self.frozen_comp5_aux_encoder.eval()
            with torch.no_grad():
                frozen_encoded = self.frozen_comp5_aux_encoder(frozen_input)
                emb5_frozen = self.pooling_fun(
                    frozen_encoded.x, frozen_encoded.batch)
            if emb5_frozen.ndim != 2 or emb5_frozen.size(-1) != self.hidden_dim:
                raise RuntimeError(
                    'Frozen Comp5 auxiliary embedding shape mismatch: '
                    f'{tuple(emb5_frozen.shape)}; expected [batch, {self.hidden_dim}]'
                )
        data5_encoded = self.comp5_encoder(data5)
        emb5 = self.pooling_fun(data5_encoded.x, data5_encoded.batch)  # [B, hidden_dim]
        if cfg.use_fifth_identity_embedding:
            if not hasattr(data5, 'component_vocab_id'):
                raise ValueError('Fifth identity embedding requires component_vocab_id in the data loader.')
            fifth_ids = data5.component_vocab_id.view(-1).long()
            emb5 = emb5 + self.fifth_component_embedding(fifth_ids)
        if cfg.use_fifth_class_embedding:
            if not hasattr(data5, 'fifth_class_id'):
                raise ValueError(
                    'Fifth-class embedding requires fifth_class_id in the data loader.')
            class_ids = data5.fifth_class_id.view(-1).long()
            invalid = (
                (class_ids < 0)
                | (class_ids >= self.fifth_class_embedding.num_embeddings)
            )
            if invalid.any():
                raise IndexError(
                    'Fifth-class vocabulary ID exceeds the configured '
                    'embedding table.')
            emb5 = emb5 + self.fifth_class_embedding(class_ids)
        if component_aux_enabled(cfg, 4):
            aux_feat = data5.aux_feat.view(data5.num_graphs, -1).float()
            if aux_feat.size(1) != cfg.component_aux_dim:
                raise ValueError(
                    f"Expected {cfg.component_aux_dim} auxiliary features, "
                    f"got {aux_feat.size(1)}.")
            emb5 = emb5 + self.aux_feature_encoder(aux_feat)
        ratio5 = data5.ratio.view(-1, 1).float()
        if self.use_fifth_structured_features:
            aa_id = data5.fifth_aa_id.view(-1).long(); terminal_id = data5.fifth_terminal_id.view(-1).long()
            if aa_id.max() >= self.fifth_aa_embedding.num_embeddings or terminal_id.max() >= self.fifth_terminal_embedding.num_embeddings:
                raise IndexError('O13G structured categorical ID exceeds its train-derived vocabulary.')
            tail = data5.fifth_tail_feat.view(data5.num_graphs, 2).float()
            aa = self.fifth_aa_embedding(aa_id)
            structured = torch.cat([aa, tail[:, :1], tail[:, 1:2], self.fifth_terminal_embedding(terminal_id), aa * tail[:, :1]], dim=1)
            emb5 = emb5 + self.fifth_structured_encoder(structured) * (ratio5 > 0).float()
        if self.fifth_semantic_feature_dim:
            semantic = data5.fifth_semantic_feat.view(data5.num_graphs, -1).float()
            if semantic.size(1) != self.fifth_semantic_feature_dim:
                raise ValueError('Fifth semantic feature dimension does not match configuration.')
            # The MLP has biases, so preserve [Fr]/ratio=0 absence semantics
            # explicitly rather than allowing an all-zero feature vector to
            # create a learned absent-component offset.
            emb5 = emb5 + self.fifth_semantic_encoder(semantic) * (ratio5 > 0).float()
        if cfg.use_fifth_ratio_modulation:
            ratio5_features = self._build_ratio_feature(ratio5)
            gamma, beta = self.fifth_ratio_modulator(ratio5_features).chunk(2, dim=-1)
            emb5 = emb5 * (1.0 + torch.tanh(gamma)) + beta

        # --- Fusion ---
        all_ratios = torch.cat(
            [ratio1, ratio2, ratio3, ratio4, ratio5], dim=1)
        all_embs = torch.cat([emb1, emb2, emb3, emb4, emb5], dim=1)
        if self.fifth_only_fusion:
            fusion_parts = [emb5, ratio5]
            if cfg.use_mordred_features:
                fusion_parts.append(
                    data5.mordred_feat.view(
                        data5.num_graphs, -1).float())
            if self.fifth_mechanistic_descriptor_dim:
                fifth_descriptor = data5.fifth_mechanistic_feat.view(data5.num_graphs, -1).float()
                if fifth_descriptor.size(1) != self.fifth_mechanistic_descriptor_dim:
                    raise ValueError('Fifth mechanistic descriptor dimension does not match configuration.')
                fusion_parts.append(fifth_descriptor)
        else:
            fusion_parts = [all_embs, all_ratios]
            if self.ratio_polynomial_features:
                fusion_parts.append(
                    self._build_composition_basis(all_ratios))
            if cfg.use_mordred_features:
                mordred_batches = (
                    [data5]
                    if cfg.mordred_fifth_only
                    else [data1, data2, data3, data4, data5]
                )
                mordred_input = torch.cat([
                    data.mordred_feat.view(
                        data.num_graphs, -1).float()
                    for data in mordred_batches
                ], dim=1)
                fusion_parts.append(mordred_input)
            if self.fifth_mechanistic_descriptor_dim:
                fifth_descriptor = data5.fifth_mechanistic_feat.view(data5.num_graphs, -1).float()
                if fifth_descriptor.size(1) != self.fifth_mechanistic_descriptor_dim:
                    raise ValueError('Fifth mechanistic descriptor dimension does not match configuration.')
                fusion_parts.append(fifth_descriptor)
        if self.frozen_comp5_aux_enable:
            if emb5_frozen is None:
                raise RuntimeError(
                    'Frozen Comp5 auxiliary branch was enabled but did not produce an embedding.')
            # Keep fusion_parts[0] as the five component tokens; gated and
            # attention fusion rewrite that slot later in this method.
            fusion_parts.insert(1, emb5_frozen)
        combined = torch.cat(fusion_parts, dim=1)
        if self.fusion_type == 'gated_concat':
            gates = self.component_gate(combined).unsqueeze(-1)
            gated_embs = (all_embs.view(-1, self.num_components, self.hidden_dim) * gates)
            fusion_parts[0] = gated_embs.reshape(gated_embs.size(0), -1)
            combined = torch.cat(fusion_parts, dim=1)
        elif self.fusion_type == 'attention_concat':
            tokens = all_embs.view(-1, self.num_components, self.hidden_dim)
            attended, _ = self.component_attention(
                tokens + self.component_position, tokens + self.component_position,
                tokens, need_weights=False)
            tokens = self.component_attention_norm(tokens + attended)
            fusion_parts[0] = tokens.reshape(tokens.size(0), -1)
            combined = torch.cat(fusion_parts, dim=1)
        threshold_logit = None
        if self.use_norm_threshold_head:
            fused = self.fusion_backbone(combined)
            pred = self.regression_head(fused)
            threshold_logit = self.norm_threshold_head(fused)
            if self.fusion_type == 'residual_concat':
                pred = pred + self.fusion_linear_skip(combined)
        elif cfg.model.target_specific_heads:
            fused = self.fusion_backbone(combined)
            pred = torch.cat([head(fused) for head in self.target_heads], dim=1)
        else:
            pred = self.fusion(combined)  # [B, out_dim]
            if self.fusion_type == 'residual_concat':
                pred = pred + self.fusion_linear_skip(combined)
        if self.output_activation == 'sigmoid':
            pred = torch.sigmoid(pred)

        data1.graph_feature = pred
        pred, label = self._apply_index(data1)
        pred = pred.contiguous().view(-1)
        label = label.contiguous().view(-1)
        if threshold_logit is not None:
            return pred, label, threshold_logit.contiguous().view(-1)
        return pred, label
