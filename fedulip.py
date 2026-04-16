import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Iterable, List, Optional, Sequence

from utils.tokenizer import SimpleTokenizer


DEFAULT_PROMPT_TEMPLATE = "a point cloud of {}."


class InversionNetwork(nn.Module):
    """
    [Local-Private] Maps raw 3D features to CLIP word embedding space.
    Reference: PromptSG. Purpose: Capture client-specific data distribution.
    """
    def __init__(self, feature_dim: int = 512, token_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, feature_dim]
        # return: [Batch, 1, token_dim] (Pseudo token S*)
        return self.mlp(x).unsqueeze(1)


class VisualAdapter(nn.Module):
    """
    [Shared-Global] Gated adapter for frozen 3D backbone.
    Reference: FedCLIP. Purpose: Parameter-efficient fine-tuning.
    """
    def __init__(self, dim: int = 768, ratio: int = 1):
        super().__init__()
        hidden_dim = max(1, dim // ratio)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.adapter(x)


class SACA(nn.Module):
    """
    [Shared-Global] Cross-Modal Attention module.
    Reference: FedMVP. Purpose: Semantics-guided geometry refinement.
    Direction: Query=Text_Feats, Key/Value=3D_Feats.
    """
    def __init__(self, dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_3d: torch.Tensor, feat_text: torch.Tensor) -> torch.Tensor:
        # Text query guides which 3D tokens to attend under domain shift.
        # feat_3d: [Batch, 1, Dim], feat_text: [Batch, 1, Dim]
        attn_out, _ = self.attn(query=feat_text, key=feat_3d, value=feat_3d, need_weights=False)
        return self.norm(feat_3d + attn_out)


class DynamicPromptEncoder(nn.Module):
    """
    Builds CLIP-like text embeddings with a style token injected before the class name.
    The style token is inserted by shifting token embeddings so the prompt reads like:
    "A 3D model of a [S*] [CLASS]".
    """
    def __init__(
        self,
        token_embedding: nn.Embedding,
        positional_embedding: torch.Tensor,
        transformer: nn.Module,
        ln_final: nn.Module,
        text_projection: torch.Tensor,
        context_length: int = 77,
        tokenizer: Optional[SimpleTokenizer] = None,
        prompt_templates: Optional[Sequence[str]] = None,
        prompt_template: Optional[str] = None,
        prompt_chunk_size: int = 16,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection
        self.context_length = context_length
        self.tokenizer = tokenizer or SimpleTokenizer()
        self._cached_token_map = {}
        if prompt_templates:
            self.prompt_templates = list(prompt_templates)
        elif prompt_template:
            self.prompt_templates = [prompt_template]
        else:
            self.prompt_templates = [DEFAULT_PROMPT_TEMPLATE]
        self.prompt_chunk_size = max(1, int(prompt_chunk_size))

    def _find_subsequence_start(
        self,
        tokens: Sequence[int],
        pattern: Sequence[int],
        end_idx: int,
    ) -> Optional[int]:
        if not pattern:
            return None
        max_start = max(0, end_idx - len(pattern))
        for start in range(0, max_start + 1):
            if list(tokens[start:start + len(pattern)]) == list(pattern):
                return start
        return None

    def _insert_style_before_class(
        self,
        token_embeddings: torch.Tensor,
        style_vec: torch.Tensor,
        tokens: torch.Tensor,
        class_token_ids: Sequence[int],
        eot_idx: int,
    ) -> torch.Tensor:
        tokens_list = tokens.tolist()
        class_start = self._find_subsequence_start(tokens_list, class_token_ids, int(eot_idx))
        if class_start is None or class_start >= int(eot_idx):
            replace_idx = max(int(eot_idx) - 1, 0)
            token_embeddings[replace_idx] = style_vec
            return token_embeddings

        class_end = class_start + len(class_token_ids)
        if class_end >= int(eot_idx):
            replace_idx = class_start - 1 if class_start > 0 else max(int(eot_idx) - 1, 0)
            token_embeddings[replace_idx] = style_vec
            return token_embeddings

        shifted = token_embeddings.clone()
        shift_end = int(eot_idx)
        shifted[class_start + 1:shift_end + 1] = token_embeddings[class_start:shift_end]
        shifted[class_start] = style_vec
        shifted[shift_end] = token_embeddings[shift_end]
        return shifted

    @classmethod
    def from_ulip_model(
        cls,
        ulip_model: nn.Module,
        tokenizer: Optional[SimpleTokenizer] = None,
        prompt_templates: Optional[Sequence[str]] = None,
        prompt_template: Optional[str] = None,
        prompt_chunk_size: int = 16,
    ) -> "DynamicPromptEncoder":
        if not hasattr(ulip_model, "token_embedding"):
            raise ValueError("ULIP model does not expose CLIP text components.")
        return cls(
            token_embedding=ulip_model.token_embedding,
            positional_embedding=ulip_model.positional_embedding,
            transformer=ulip_model.transformer,
            ln_final=ulip_model.ln_final,
            text_projection=ulip_model.text_projection,
            context_length=ulip_model.context_length,
            tokenizer=tokenizer,
            prompt_templates=prompt_templates,
            prompt_template=prompt_template,
            prompt_chunk_size=prompt_chunk_size,
        )

    def _build_prompts_for_class(self, class_name: str) -> List[str]:
        return [t.format(class_name) for t in self.prompt_templates]

    def forward(
        self,
        class_names: Sequence[str],
        style_token: Optional[torch.Tensor],
        prompt_templates: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if isinstance(class_names, str):
            class_names = [class_names]

        if prompt_templates is None:
            active_templates = self.prompt_templates
        elif isinstance(prompt_templates, str):
            active_templates = [prompt_templates]
        else:
            active_templates = list(prompt_templates)

        device = style_token.device if style_token is not None else self.token_embedding.weight.device
        if style_token is not None and style_token.dim() == 3:
            style_token = style_token.squeeze(1)

        text_features = []
        for idx, name in enumerate(class_names):
            prompts = [t.format(name) for t in active_templates]
            tokens = self.tokenizer(prompts, context_length=self.context_length)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            tokens = tokens.to(device)

            if style_token is not None:
                style_vec = style_token[idx]
                if style_vec.dim() != 1:
                    raise ValueError("style_token must be [Batch, TokenDim].")
            else:
                style_vec = None
            if style_vec is not None and name not in self._cached_token_map:
                self._cached_token_map[name] = self.tokenizer.encode(name)
            class_token_ids = self._cached_token_map.get(name, [])

            prompt_feats = []
            for start in range(0, tokens.shape[0], self.prompt_chunk_size):
                end = start + self.prompt_chunk_size
                tokens_chunk = tokens[start:end]
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    token_embeddings = self.token_embedding(tokens_chunk)

                    eot_indices = tokens_chunk.argmax(dim=-1)
                    if style_vec is not None:
                        if style_vec.shape[-1] != token_embeddings.shape[-1]:
                            raise ValueError("style_token dim does not match text token width.")
                        style_vec_chunk = style_vec.to(token_embeddings.dtype)
                        style_vec_chunk = style_vec_chunk.unsqueeze(0).expand(token_embeddings.shape[0], -1)
                        updated = []
                        for row_idx in range(token_embeddings.shape[0]):
                            updated.append(
                                self._insert_style_before_class(
                                    token_embeddings[row_idx],
                                    style_vec_chunk[row_idx],
                                    tokens_chunk[row_idx],
                                    class_token_ids,
                                    eot_indices[row_idx].item(),
                                )
                            )
                        token_embeddings = torch.stack(updated, dim=0)

                    pos_embed = self.positional_embedding.to(
                        device=token_embeddings.device,
                        dtype=token_embeddings.dtype,
                    )
                    x = token_embeddings + pos_embed
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.ln_final(x)

                    chunk_feat = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection
                    prompt_feats.append(F.normalize(chunk_feat, dim=-1))

            prompt_feats = torch.cat(prompt_feats, dim=0)
            class_feat = prompt_feats.mean(dim=0)
            class_feat = F.normalize(class_feat, dim=-1)
            text_features.append(class_feat)

        return torch.stack(text_features, dim=0)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        style_token: Optional[torch.Tensor],
        class_token_ids: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if style_token is not None and style_token.dim() == 3:
            style_token = style_token.squeeze(1)

        token_embeddings = self.token_embedding(tokens)
        eot_indices = tokens.argmax(dim=-1)
        if style_token is not None:
            if style_token.shape[0] != tokens.shape[0]:
                raise ValueError("style_token batch size does not match tokens.")
            if style_token.shape[-1] != token_embeddings.shape[-1]:
                raise ValueError("style_token dim does not match text token width.")
            style_vec = style_token.to(token_embeddings.dtype)
            if class_token_ids is not None:
                if len(class_token_ids) != tokens.shape[0]:
                    raise ValueError("class_token_ids batch size does not match tokens.")
                updated = []
                for row_idx in range(tokens.shape[0]):
                    updated.append(
                        self._insert_style_before_class(
                            token_embeddings[row_idx],
                            style_vec[row_idx],
                            tokens[row_idx],
                            class_token_ids[row_idx],
                            eot_indices[row_idx].item(),
                        )
                    )
                token_embeddings = torch.stack(updated, dim=0)
            else:
                replace_indices = (eot_indices - 1).clamp(min=0)
                replace_mask = F.one_hot(replace_indices, num_classes=token_embeddings.shape[1])
                replace_mask = replace_mask.to(token_embeddings.dtype).unsqueeze(-1)
                token_embeddings = (
                    token_embeddings * (1.0 - replace_mask)
                    + style_vec.unsqueeze(1) * replace_mask
                )

        pos_embed = self.positional_embedding.to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        x = token_embeddings + pos_embed
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection
        return x


class FedULIPClient(nn.Module):
    """
    Federated client with private inversion network and shared adapter/SACA modules.
    """
    def __init__(
        self,
        point_encoder: nn.Module,
        text_encoder: DynamicPromptEncoder,
        inversion_net: InversionNetwork,
        adapter: VisualAdapter,
        saca: SACA,
        pc_projection: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
        train_loader: Optional[Iterable] = None,
        label_names: Optional[Sequence[str]] = None,
        optimizer_cls=torch.optim.AdamW,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        eval_chunk_size: int = 512,
        device: Optional[torch.device] = None,
        use_inversion: bool = True,
        share_inversion: bool = False,
        use_saca: bool = True,
        pseudo_token_mode: str = "instance",
        supcon_weight: float = 0.1,
        supcon_mode: str = "point_text",
        triplet_weight: float = 0.0,
        triplet_margin: float = 0.3,
    ):
        super().__init__()
        self.point_encoder = point_encoder
        self.text_encoder = text_encoder
        self.inversion_net = inversion_net
        self.adapter = adapter
        self.saca = saca
        self.pc_projection = pc_projection
        self.logit_scale = logit_scale
        self.temperature = temperature
        self.train_loader = train_loader
        self.label_names = list(label_names) if label_names is not None else None
        self.eval_chunk_size = max(1, int(eval_chunk_size))
        self.eval_class_names = None
        self.eval_class_text_feats = None
        self.use_inversion = use_inversion
        self.share_inversion = share_inversion
        self.use_saca = use_saca
        self.pseudo_token_mode = pseudo_token_mode
        self.supcon_weight = supcon_weight
        self.supcon_mode = supcon_mode
        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}

        if self.pseudo_token_mode not in ("class", "instance"):
            raise ValueError("pseudo_token_mode must be 'class' or 'instance'.")
        if self.supcon_mode not in ("point", "point_text"):
            raise ValueError("supcon_mode must be 'point' or 'point_text'.")

        self.freeze_backbones()
        if device is not None:
            self.to(device)
        self.optimizer = optimizer_cls(self._trainable_parameters(), lr=lr, weight_decay=weight_decay)

    @classmethod
    def from_ulip_model(
        cls,
        ulip_model: nn.Module,
        adapter_ratio: int = 1,
        saca_heads: int = 8,
        inversion_hidden_dim: int = 512,
        prompt_templates: Optional[Sequence[str]] = None,
        prompt_template: Optional[str] = None,
        prompt_chunk_size: int = 16,
        eval_chunk_size: int = 512,
        tokenizer: Optional[SimpleTokenizer] = None,
        use_inversion: bool = True,
        share_inversion: bool = False,
        use_saca: bool = True,
        pseudo_token_mode: str = "instance",
        supcon_weight: float = 0.1,
        supcon_mode: str = "point_text",
        triplet_weight: float = 0.0,
        triplet_margin: float = 0.3,
        **kwargs,
    ) -> "FedULIPClient":
        if not hasattr(ulip_model, "pc_projection"):
            raise ValueError("ULIP model must expose pc_projection for FedULIP.")
        pc_proj = ulip_model.pc_projection
        embed_dim = pc_proj.shape[1]
        token_dim = ulip_model.token_embedding.weight.shape[1]

        text_encoder = DynamicPromptEncoder.from_ulip_model(
            ulip_model,
            tokenizer=tokenizer,
            prompt_templates=prompt_templates,
            prompt_template=prompt_template,
            prompt_chunk_size=prompt_chunk_size,
        )
        inversion_net = InversionNetwork(
            feature_dim=embed_dim,
            token_dim=token_dim,
            hidden_dim=inversion_hidden_dim,
        )
        adapter = VisualAdapter(dim=embed_dim, ratio=adapter_ratio)
        saca = SACA(dim=embed_dim, num_heads=saca_heads)

        return cls(
            point_encoder=ulip_model.point_encoder,
            text_encoder=text_encoder,
            inversion_net=inversion_net,
            adapter=adapter,
            saca=saca,
            pc_projection=pc_proj,
            logit_scale=getattr(ulip_model, "logit_scale", None),
            eval_chunk_size=eval_chunk_size,
            use_inversion=use_inversion,
            share_inversion=share_inversion,
            use_saca=use_saca,
            pseudo_token_mode=pseudo_token_mode,
            supcon_weight=supcon_weight,
            supcon_mode=supcon_mode,
            triplet_weight=triplet_weight,
            triplet_margin=triplet_margin,
            **kwargs,
        )

    def freeze_backbones(self) -> None:
        for module in [self.point_encoder, self.text_encoder]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        if isinstance(self.pc_projection, nn.Parameter):
            self.pc_projection.requires_grad = False
        if isinstance(self.logit_scale, nn.Parameter):
            self.logit_scale.requires_grad = True

    def enable_data_parallel(self, device_ids: Sequence[int]) -> None:
        if not torch.cuda.is_available():
            return
        device_ids = [int(dev_id) for dev_id in device_ids if dev_id is not None]
        if len(device_ids) < 2:
            return
        self.point_encoder = nn.DataParallel(self.point_encoder, device_ids=device_ids)
        self.adapter = nn.DataParallel(self.adapter, device_ids=device_ids)
        self.inversion_net = nn.DataParallel(self.inversion_net, device_ids=device_ids)
        self.saca = nn.DataParallel(self.saca, device_ids=device_ids)
        self.optimizer = self.optimizer_cls(self._trainable_parameters(), **self.optimizer_kwargs)

    def _trainable_parameters(self) -> List[nn.Parameter]:
        params = list(self.adapter.parameters())
        if self.use_inversion:
            params += list(self.inversion_net.parameters())
        if self.use_saca:
            params += list(self.saca.parameters())
        if isinstance(self.logit_scale, nn.Parameter) and self.logit_scale.requires_grad:
            params.append(self.logit_scale)
        return params

    def _project_pc(self, raw_3d: torch.Tensor) -> torch.Tensor:
        if self.pc_projection is None:
            return raw_3d
        if isinstance(self.pc_projection, nn.Module):
            return self.pc_projection(raw_3d)
        return raw_3d @ self.pc_projection

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_scale is not None:
            return logits * self.logit_scale.exp()
        return logits / self.temperature

    def _labels_to_names(self, labels: torch.Tensor) -> List[str]:
        if self.label_names is None:
            return [str(int(x)) for x in labels]
        return [self.label_names[int(x)] for x in labels]

    def _unpack_batch(self, batch) -> Dict[str, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            points, labels, label_names = batch
            class_names = list(label_names)
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            points, labels = batch
            class_names = self._labels_to_names(labels)
        else:
            raise ValueError("Batch must be (points, labels) or (points, labels, label_names).")
        return {"points": points, "labels": labels, "class_names": class_names}

    def get_text_features(
        self,
        class_names: Sequence[str],
        style_token: Optional[torch.Tensor],
        is_training: Optional[bool] = None,
    ) -> torch.Tensor:
        if is_training is None:
            is_training = self.training
        if is_training:
            templates = self.text_encoder.prompt_templates
            if not templates:
                raise ValueError("Prompt templates are not initialized.")
            template = random.choice(templates)
            text_feats = self.text_encoder(class_names, style_token, prompt_templates=[template])
            dtype = style_token.dtype if style_token is not None else text_feats.dtype
            return F.normalize(text_feats, dim=-1).to(dtype)
        if style_token is None:
            return self.get_class_text_features_eval(class_names)
        return self.get_text_features_eval(class_names, style_token)

    def get_text_features_instance(
        self,
        class_names: Sequence[str],
        style_token: torch.Tensor,
        is_training: Optional[bool] = None,
    ) -> torch.Tensor:
        if is_training is None:
            is_training = self.training
        templates = self.text_encoder.prompt_templates
        if not templates:
            raise ValueError("Prompt templates are not initialized.")
        if isinstance(class_names, str):
            class_names = [class_names]

        if style_token.dim() == 3:
            style_token = style_token.squeeze(1)
        device = style_token.device
        name_list = list(class_names)
        if is_training:
            template_list = [random.choice(templates)]
        else:
            template_list = list(templates)

        bsz = style_token.shape[0]
        num_classes = len(name_list)
        num_templates = len(template_list)
        total = bsz * num_classes * num_templates
        class_token_ids_map = {
            name: self.text_encoder.tokenizer.encode(name) for name in name_list
        }

        feats = []
        chunk_size = self.eval_chunk_size
        context = torch.enable_grad() if is_training else torch.inference_mode()
        with context:
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                flat_indices = torch.arange(start, end, device=device)
                sample_indices = flat_indices // (num_classes * num_templates)
                rem = flat_indices % (num_classes * num_templates)
                class_indices = rem // num_templates
                template_indices = rem % num_templates

                class_idx_list = class_indices.tolist()
                template_idx_list = template_indices.tolist()
                prompts_chunk = [
                    template_list[t_idx].format(name_list[c_idx])
                    for c_idx, t_idx in zip(class_idx_list, template_idx_list)
                ]
                class_token_ids_chunk = [
                    class_token_ids_map[name_list[c_idx]] for c_idx in class_idx_list
                ]
                tokens_chunk = self.text_encoder.tokenizer(
                    prompts_chunk,
                    context_length=self.text_encoder.context_length,
                )
                if tokens_chunk.dim() == 1:
                    tokens_chunk = tokens_chunk.unsqueeze(0)
                tokens_chunk = tokens_chunk.to(device)
                style_chunk = style_token[sample_indices]

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    chunk_feat = self.text_encoder.encode_tokens(
                        tokens_chunk, style_chunk, class_token_ids_chunk
                    )
                feats.append(F.normalize(chunk_feat, dim=-1))

        all_feats = torch.cat(feats, dim=0)
        all_feats = all_feats.view(bsz, num_classes, num_templates, -1).mean(dim=2)
        return F.normalize(all_feats, dim=-1).to(style_token.dtype)

    def get_text_features_eval(
        self,
        class_names: Sequence[str],
        style_token: torch.Tensor,
    ) -> torch.Tensor:
        templates = self.text_encoder.prompt_templates
        if not templates:
            raise ValueError("Prompt templates are not initialized.")
        if isinstance(class_names, str):
            class_names = [class_names]

        if style_token.dim() == 3:
            style_token = style_token.squeeze(1)

        device = style_token.device
        name_list = list(class_names)
        template_list = list(templates)
        bsz = len(name_list)
        num_templates = len(template_list)
        total = bsz * num_templates
        class_token_ids_map = {
            name: self.text_encoder.tokenizer.encode(name) for name in name_list
        }

        feats = []
        chunk_size = self.eval_chunk_size
        with torch.inference_mode():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                flat_indices = torch.arange(start, end, device=device)
                batch_indices = flat_indices // num_templates
                template_indices = flat_indices % num_templates

                batch_idx_list = batch_indices.tolist()
                template_idx_list = template_indices.tolist()
                prompts_chunk = [
                    template_list[t_idx].format(name_list[b_idx])
                    for b_idx, t_idx in zip(batch_idx_list, template_idx_list)
                ]
                class_token_ids_chunk = [
                    class_token_ids_map[name_list[b_idx]] for b_idx in batch_idx_list
                ]
                tokens_chunk = self.text_encoder.tokenizer(
                    prompts_chunk,
                    context_length=self.text_encoder.context_length,
                )
                if tokens_chunk.dim() == 1:
                    tokens_chunk = tokens_chunk.unsqueeze(0)
                tokens_chunk = tokens_chunk.to(device)
                style_chunk = style_token[batch_indices]

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    chunk_feat = self.text_encoder.encode_tokens(
                        tokens_chunk, style_chunk, class_token_ids_chunk
                    )
                feats.append(F.normalize(chunk_feat, dim=-1))

        all_feats = torch.cat(feats, dim=0)
        all_feats = all_feats.view(bsz, num_templates, -1).mean(dim=1)
        return F.normalize(all_feats, dim=-1).to(style_token.dtype)

    def get_class_text_features_eval(self, class_names: Sequence[str]) -> torch.Tensor:
        templates = self.text_encoder.prompt_templates
        if not templates:
            raise ValueError("Prompt templates are not initialized.")
        if isinstance(class_names, str):
            class_names = [class_names]

        name_list = list(class_names)
        template_list = list(templates)
        num_classes = len(name_list)
        num_templates = len(template_list)
        total = num_classes * num_templates
        device = self.text_encoder.token_embedding.weight.device

        feats = []
        chunk_size = self.eval_chunk_size
        with torch.inference_mode():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                flat_indices = torch.arange(start, end, device=device)
                class_indices = flat_indices // num_templates
                template_indices = flat_indices % num_templates

                class_idx_list = class_indices.tolist()
                template_idx_list = template_indices.tolist()
                prompts_chunk = [
                    template_list[t_idx].format(name_list[c_idx])
                    for c_idx, t_idx in zip(class_idx_list, template_idx_list)
                ]
                tokens_chunk = self.text_encoder.tokenizer(
                    prompts_chunk,
                    context_length=self.text_encoder.context_length,
                )
                if tokens_chunk.dim() == 1:
                    tokens_chunk = tokens_chunk.unsqueeze(0)
                tokens_chunk = tokens_chunk.to(device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    chunk_feat = self.text_encoder.encode_tokens(tokens_chunk, style_token=None)
                feats.append(F.normalize(chunk_feat, dim=-1))

        all_feats = torch.cat(feats, dim=0)
        all_feats = all_feats.view(num_classes, num_templates, -1).mean(dim=1)
        return F.normalize(all_feats, dim=-1)

    def get_point_features(self, points: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raw_3d = self.point_encoder(points)
        return raw_3d

    def _supcon_loss(self, feats: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        if feats.shape[0] < 2:
            return None
        labels = labels.view(-1, 1)
        device = feats.device
        same_class = labels.eq(labels.T)
        diag = torch.eye(labels.shape[0], device=device, dtype=torch.bool)
        pos_mask = same_class & ~diag
        if not torch.any(pos_mask):
            return None
        feats = F.normalize(feats, dim=-1)
        sim = feats @ feats.T
        sim = sim / max(self.temperature, 1e-8)
        sim = sim - sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim) * (~diag)
        denom = exp_sim.sum(dim=1)
        numer = (exp_sim * pos_mask).sum(dim=1)
        valid = numer > 0
        if not torch.any(valid):
            return None
        loss = -torch.log((numer[valid] / (denom[valid] + 1e-12)).clamp_min(1e-12))
        return loss.mean()

    def _supcon_cross(
        self,
        anchors: torch.Tensor,
        contrasts: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if anchors.shape[0] < 1:
            return None
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T)
        if not torch.any(pos_mask):
            return None
        anchors = F.normalize(anchors, dim=-1)
        contrasts = F.normalize(contrasts, dim=-1)
        sim = anchors @ contrasts.T
        sim = sim / max(self.temperature, 1e-8)
        sim = sim - sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1)
        numer = (exp_sim * pos_mask).sum(dim=1)
        valid = numer > 0
        if not torch.any(valid):
            return None
        loss = -torch.log((numer[valid] / (denom[valid] + 1e-12)).clamp_min(1e-12))
        return loss.mean()

    def _supcon_point_text(
        self,
        point_feats: torch.Tensor,
        text_feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        loss_i2t = self._supcon_cross(point_feats, text_feats, labels)
        loss_t2i = self._supcon_cross(text_feats, point_feats, labels)
        if loss_i2t is None:
            return loss_t2i
        if loss_t2i is None:
            return loss_i2t
        return loss_i2t + loss_t2i

    def _batch_hard_triplet_loss(self, feats: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        if feats.shape[0] < 2:
            return None
        labels = labels.view(-1, 1)
        device = feats.device
        same_class = labels.eq(labels.T)
        diag = torch.eye(labels.shape[0], device=device, dtype=torch.bool)
        pos_mask = same_class & ~diag
        neg_mask = ~same_class
        if not torch.any(pos_mask) or not torch.any(neg_mask):
            return None
        feats = F.normalize(feats, dim=-1)
        sim = feats @ feats.T
        dist = 1.0 - sim
        dist_pos = dist.masked_fill(~pos_mask, float("-inf"))
        dist_neg = dist.masked_fill(~neg_mask, float("inf"))
        d_ap = dist_pos.max(dim=1).values
        d_an = dist_neg.min(dim=1).values
        valid = torch.isfinite(d_ap) & torch.isfinite(d_an)
        if not torch.any(valid):
            return None
        loss = F.relu(d_ap[valid] - d_an[valid] + self.triplet_margin)
        return loss.mean()

    def local_train(self, global_state: Dict[str, Dict[str, torch.Tensor]], local_epochs: int = 1) -> Dict:
        if self.train_loader is None:
            raise ValueError("train_loader must be set for local training.")
        device = next(self.adapter.parameters()).device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        self.load_global_state(global_state)
        self.inversion_net.train()
        self.adapter.train()
        self.saca.train()

        total_loss = 0.0
        total_ce = 0.0
        total_acc = 0.0
        total_s_star = 0.0
        total_raw = 0.0
        total_pc = 0.0
        total_supcon = 0.0
        total_triplet = 0.0
        total_batches = 0

        for _ in range(local_epochs):
            for batch in self.train_loader:
                payload = self._unpack_batch(batch)
                points = payload["points"].to(next(self.adapter.parameters()).device)
                class_names = payload["class_names"]
                labels = payload["labels"].to(points.device)

                raw_3d = self.get_point_features(points)
                pc_embed = self._project_pc(raw_3d)
                adapted_3d = self.adapter(pc_embed)
                s_star = None
                s_star_flat = None
                if self.use_inversion:
                    s_star = self.inversion_net(adapted_3d)
                    s_star_flat = s_star.squeeze(1)
                unique_labels, inverse = labels.unique(sorted=True, return_inverse=True)
                if self.label_names is not None:
                    class_names_unique = [self.label_names[int(idx)] for idx in unique_labels.tolist()]
                else:
                    label_to_name = {}
                    for lbl, name in zip(labels.tolist(), class_names):
                        label_to_name.setdefault(int(lbl), name)
                    class_names_unique = [label_to_name[int(idx)] for idx in unique_labels.tolist()]

                use_instance = self.use_inversion and self.pseudo_token_mode == "instance"
                if use_instance:
                    text_feats_all = self.get_text_features_instance(
                        class_names_unique,
                        s_star_flat,
                        is_training=True,
                    ).to(raw_3d.dtype)
                    text_feats = text_feats_all[torch.arange(labels.shape[0], device=labels.device), inverse]
                elif self.use_inversion:
                    class_style = torch.zeros(
                        unique_labels.shape[0],
                        s_star_flat.shape[1],
                        device=s_star_flat.device,
                        dtype=s_star_flat.dtype,
                    )
                    class_style.index_add_(0, inverse, s_star_flat)
                    counts = torch.bincount(inverse, minlength=unique_labels.shape[0]).to(s_star_flat.device)
                    class_style = class_style / counts.unsqueeze(1).clamp(min=1)
                    text_feats_all = self.get_text_features(
                        class_names_unique,
                        class_style,
                        is_training=True,
                    ).to(raw_3d.dtype)
                else:
                    text_feats_all = self.get_text_features(
                        class_names_unique,
                        None,
                        is_training=True,
                    ).to(raw_3d.dtype)
                if not use_instance:
                    text_feats = text_feats_all[inverse]

                if self.use_saca:
                    final_3d = self.saca(adapted_3d.unsqueeze(1), text_feats.unsqueeze(1)).squeeze(1)
                else:
                    final_3d = adapted_3d

                final_3d = F.normalize(final_3d, dim=-1)
                if use_instance:
                    logits = self._scale_logits(torch.sum(final_3d.unsqueeze(1) * text_feats_all, dim=-1))
                else:
                    logits = self._scale_logits(final_3d @ text_feats_all.T)
                ce_loss = F.cross_entropy(logits, inverse)

                supcon_loss = None
                triplet_loss = None
                supcon_weight = self.supcon_weight if self.use_inversion else 0.0
                if supcon_weight > 0:
                    if self.supcon_mode == "point_text":
                        supcon_loss = self._supcon_point_text(final_3d, text_feats, inverse)
                    else:
                        supcon_loss = self._supcon_loss(final_3d, inverse)
                if self.triplet_weight > 0:
                    triplet_loss = self._batch_hard_triplet_loss(final_3d, inverse)

                loss = ce_loss
                if supcon_loss is not None:
                    loss = loss + supcon_weight * supcon_loss
                if triplet_loss is not None:
                    loss = loss + self.triplet_weight * triplet_loss

                with torch.no_grad():
                    total_loss += float(loss.item())
                    total_ce += float(ce_loss.item())
                    total_acc += float((logits.argmax(dim=1) == inverse).float().mean().item())
                    if s_star is not None:
                        total_s_star += float(s_star.norm(dim=-1).mean().item())
                    total_raw += float(raw_3d.norm(dim=-1).mean().item())
                    total_pc += float(pc_embed.norm(dim=-1).mean().item())
                    if supcon_loss is not None:
                        total_supcon += float(supcon_loss.item())
                    if triplet_loss is not None:
                        total_triplet += float(triplet_loss.item())
                    total_batches += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        stats = {}
        if total_batches > 0:
            stats = {
                "avg_loss": total_loss / total_batches,
                "avg_ce": total_ce / total_batches,
                "avg_acc": total_acc / total_batches,
                "avg_s_star_norm": total_s_star / total_batches if self.use_inversion else 0.0,
                "avg_raw_norm": total_raw / total_batches,
                "avg_pc_norm": total_pc / total_batches,
                "avg_supcon": total_supcon / total_batches if self.supcon_weight > 0 else 0.0,
                "avg_triplet": total_triplet / total_batches if self.triplet_weight > 0 else 0.0,
            }
        if self.logit_scale is not None:
            stats["logit_scale"] = float(self.logit_scale.detach().exp().item())
        update = self.get_shareable_state()
        update["stats"] = stats
        return update

    def get_shareable_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        adapter_state = {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}
        saca_state = {k: v.detach().cpu() for k, v in self.saca.state_dict().items()}
        state = {"adapter": adapter_state, "saca": saca_state}
        if self.share_inversion and self.use_inversion:
            state["inversion"] = {k: v.detach().cpu() for k, v in self.inversion_net.state_dict().items()}
        if self.logit_scale is not None:
            state["logit_scale"] = self.logit_scale.detach().cpu()
        return state

    def load_global_state(self, global_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if "adapter" in global_state:
            self.adapter.load_state_dict(global_state["adapter"])
        if "saca" in global_state:
            self.saca.load_state_dict(global_state["saca"])
        if self.share_inversion and self.use_inversion and "inversion" in global_state:
            self.inversion_net.load_state_dict(global_state["inversion"])
        if "logit_scale" in global_state and self.logit_scale is not None:
            value = global_state["logit_scale"].to(self.logit_scale.device)
            if isinstance(self.logit_scale, nn.Parameter):
                self.logit_scale.data.copy_(value)
            else:
                self.logit_scale = value
        self.eval_class_names = None
        self.eval_class_text_feats = None

    def set_eval_class_text_features(
        self,
        class_names: Sequence[str],
        class_text_feats: torch.Tensor,
    ) -> None:
        self.eval_class_names = list(class_names)
        self.eval_class_text_feats = class_text_feats.detach()

    def zero_shot_logits(self, points: torch.Tensor, candidate_classes: Sequence[str]) -> torch.Tensor:
        self.eval()
        with torch.inference_mode():
            raw_3d = self.get_point_features(points)
            pc_embed = self._project_pc(raw_3d)
            adapted_3d = self.adapter(pc_embed)
            use_instance = self.use_inversion and self.pseudo_token_mode == "instance"
            if use_instance:
                s_star = self.inversion_net(adapted_3d).squeeze(1)
                text_feats_all = self.get_text_features_instance(
                    candidate_classes,
                    s_star,
                    is_training=False,
                ).to(adapted_3d.dtype)
            else:
                if self.eval_class_text_feats is not None and self.eval_class_names == list(candidate_classes):
                    text_feats_all = self.eval_class_text_feats
                else:
                    text_feats_all = self.get_class_text_features_eval(candidate_classes)
                text_feats_all = text_feats_all.to(adapted_3d.dtype)
            scores = []
            class_dim = 1 if use_instance else 0
            for class_idx in range(text_feats_all.shape[class_dim]):
                if use_instance:
                    t_feat = text_feats_all[:, class_idx, :]
                else:
                    t_feat = text_feats_all[class_idx].unsqueeze(0).expand(points.shape[0], -1)
                if self.use_saca:
                    refined_3d = self.saca(adapted_3d.unsqueeze(1), t_feat.unsqueeze(1)).squeeze(1)
                else:
                    refined_3d = adapted_3d
                sim = torch.sum(F.normalize(refined_3d, dim=-1) * t_feat, dim=-1)
                scores.append(sim)

            logits = torch.stack(scores, dim=1)
            return self._scale_logits(logits)

    def zero_shot_inference(self, points: torch.Tensor, candidate_classes: Sequence[str]) -> torch.Tensor:
        logits = self.zero_shot_logits(points, candidate_classes)
        return torch.argmax(logits, dim=1)


def fedavg_state_dict(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty.")
    if weights is None:
        weights = [1.0] * len(state_dicts)
    weight_sum = float(sum(weights))
    avg_state = {}
    for key in state_dicts[0].keys():
        acc = None
        for state, weight in zip(state_dicts, weights):
            tensor = state[key].detach().float().cpu() * float(weight)
            acc = tensor if acc is None else acc + tensor
        avg_state[key] = (acc / weight_sum).to(state_dicts[0][key].dtype)
    return avg_state


class FedULIPServer:
    """
    Server-side FedAvg aggregator for shared adapter/SACA (and optional inversion) parameters.
    """
    def __init__(
        self,
        adapter: VisualAdapter,
        saca: SACA,
        logit_scale: Optional[torch.Tensor] = None,
        inversion_net: Optional[InversionNetwork] = None,
    ):
        self.adapter = adapter
        self.saca = saca
        self.logit_scale = logit_scale
        self.inversion_net = inversion_net

    def get_global_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        state = {
            "adapter": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
            "saca": {k: v.detach().cpu() for k, v in self.saca.state_dict().items()},
        }
        if self.inversion_net is not None:
            state["inversion"] = {k: v.detach().cpu() for k, v in self.inversion_net.state_dict().items()}
        if self.logit_scale is not None:
            state["logit_scale"] = self.logit_scale.detach().cpu()
        return state

    def aggregate(
        self,
        client_updates: Sequence[Dict[str, Dict[str, torch.Tensor]]],
        weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        adapter_states = [update["adapter"] for update in client_updates]
        saca_states = [update["saca"] for update in client_updates]
        if weights is None:
            weights = [1.0] * len(client_updates)
        logit_states = []
        logit_weights = []
        inversion_states = []
        inversion_weights = []
        for weight, update in zip(weights, client_updates):
            if "logit_scale" in update:
                logit_states.append({"logit_scale": update["logit_scale"]})
                logit_weights.append(weight)
            if "inversion" in update:
                inversion_states.append(update["inversion"])
                inversion_weights.append(weight)

        new_adapter = fedavg_state_dict(adapter_states, weights)
        new_saca = fedavg_state_dict(saca_states, weights)

        self.adapter.load_state_dict(new_adapter)
        self.saca.load_state_dict(new_saca)
        if inversion_states:
            if self.inversion_net is None:
                raise ValueError("Received inversion updates without a server inversion network.")
            new_inversion = fedavg_state_dict(inversion_states, inversion_weights)
            self.inversion_net.load_state_dict(new_inversion)
        if logit_states:
            new_logit = fedavg_state_dict(logit_states, logit_weights)["logit_scale"]
            if self.logit_scale is None:
                self.logit_scale = new_logit
            elif isinstance(self.logit_scale, nn.Parameter):
                self.logit_scale.data.copy_(new_logit.to(self.logit_scale.device))
            else:
                self.logit_scale = new_logit
        return self.get_global_state()
