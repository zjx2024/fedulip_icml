import argparse
import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ULIP_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "ULIP"))
if ULIP_DIR not in sys.path:
    sys.path.insert(0, ULIP_DIR)

os.chdir(ULIP_DIR)

from data.dataset_3d import customized_collate_fn
from models import ULIP_models as ulip_models
from models.fedulip import FedULIPClient, FedULIPServer
from utils.sampler import PKBatchSampler, labels_for_indices
from utils.tokenizer import SimpleTokenizer
from utils.utils import get_dataset

from domain_abcd_config import DATASET_SPECS, DOMAIN_LABELS
from domain_abcd_utils import (
    DomainDataset,
    assign_clients_to_domains,
    split_indices_randomly,
    split_indices_evenly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedULIP domain A/B/C/D experiment")
    parser.add_argument("--dataset", default="modelnet40", type=str,
                        choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--model", default="ULIP_PointBERT", type=str)
    parser.add_argument("--npoints", default=1024, type=int)
    parser.add_argument("--sparse-npoints", default=512, type=int, choices=[512])
    parser.add_argument("--use-height", action="store_true")
    parser.add_argument("--prompt", default="modelnet40_64", type=str)
    parser.add_argument("--num-clients", default=5, type=int)
    parser.add_argument("--rounds", default=50, type=int)
    parser.add_argument("--local-epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--adapter-ratio", default=1, type=int)
    parser.add_argument("--saca-heads", default=8, type=int)
    parser.add_argument("--inversion-hidden-dim", default=512, type=int)
    parser.add_argument("--prompt-chunk-size", default=4, type=int)
    parser.add_argument("--eval-chunk-size", default=512, type=int)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--supcon-weight", default=0.1, type=float)
    parser.add_argument("--supcon-mode", default="point_text", type=str,
                        choices=["point", "point_text"],
                        help="SupCon mode: point uses point-point; point_text uses point-text bi-directional")
    parser.add_argument("--triplet-weight", default=0.0, type=float)
    parser.add_argument("--triplet-margin", default=0.3, type=float)
    parser.add_argument("--pk-sampler", action="store_true", default=True,
                        help="use P×K class-balanced batch sampling")
    parser.add_argument("--no-pk-sampler", action="store_false", dest="pk_sampler",
                        help="disable P×K class-balanced batch sampling")
    parser.add_argument("--pk-classes", default=8, type=int,
                        help="number of classes per batch when using P×K sampling")
    parser.add_argument("--pk-samples", default=8, type=int,
                        help="number of samples per class when using P×K sampling")
    parser.add_argument("--pretrained", default="checkpoint_pointbert.pt", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval-client-id", default=0, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--run-name", default="", type=str)
    parser.add_argument("--disable-inversion", action="store_true")
    parser.add_argument("--share-inversion", action="store_true", default=True,
                        help="upload and aggregate the inversion network")
    parser.add_argument("--no-share-inversion", action="store_false", dest="share_inversion",
                        help="disable sharing the inversion network")
    parser.add_argument("--disable-saca", action="store_true")
    parser.add_argument("--devices", default="", type=str,
                        help="comma-separated cuda device ids (e.g. 0,1); leave empty for single GPU")
    parser.add_argument("--max-concurrent-clients", default=0, type=int,
                        help="max clients trained in parallel; 0 uses number of devices")
    parser.add_argument("--single-client-data-parallel", action="store_true",
                        help="force DataParallel when only one client and multiple CUDA devices are available")
    parser.add_argument("--disable-single-client-data-parallel", action="store_true",
                        help="disable auto DataParallel for single-client runs")
    parser.add_argument("--train-max-samples", default=0, type=int,
                        help="cap training samples (0 = use full training set)")
    parser.add_argument("--eval-max-samples", default=0, type=int,
                        help="cap evaluation samples (0 = use full test set)")
    parser.add_argument("--fold", default=-1, type=int,
                        help="run a single fold (0-based); -1 runs all folds")
    parser.add_argument("--outlier-ratio", default=0.05, type=float)
    parser.add_argument("--drop-ratio", default=0.05, type=float)
    parser.add_argument("--partial-keep-ratio", default=0.7, type=float)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str, run_name: str) -> Callable[[str], None]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() or f"fedulip_domain_abcd_{timestamp}"
    log_path = os.path.join(output_dir, f"{name}.log")

    def log(message: str) -> None:
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    log(f"Log file: {log_path}")
    return log


def build_ulip_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model_args = argparse.Namespace(
        model=args.model,
        npoints=args.npoints,
        use_height=args.use_height,
        evaluate_3d=True,
        evaluate_3d_ulip2=False,
    )
    model = getattr(ulip_models, args.model)(args=model_args)

    if args.pretrained:
        try:
            ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.pretrained, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)
    return model


def resolve_devices(device_arg: str) -> List[torch.device]:
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    if device_arg.strip() == "":
        return [torch.device("cuda:0")]
    if device_arg.strip().lower() == "cpu":
        return [torch.device("cpu")]
    ids = [token.strip() for token in device_arg.split(",") if token.strip()]
    return [torch.device(f"cuda:{dev_id}") for dev_id in ids]


def build_ulip_models(args: argparse.Namespace, devices: Sequence[torch.device]) -> Dict[str, torch.nn.Module]:
    models = {}
    for device in devices:
        models[str(device)] = build_ulip_model(args, device)
    return models


def build_dataset_args(
    args: argparse.Namespace,
    train_dataset: str,
    eval_dataset: str,
) -> argparse.Namespace:
    data_args = argparse.Namespace()
    data_args.pretrain_dataset_name = train_dataset
    data_args.validate_dataset_name = eval_dataset
    data_args.pretrain_dataset_prompt = args.prompt
    data_args.validate_dataset_prompt = args.prompt
    data_args.model = args.model
    data_args.npoints = args.npoints
    data_args.use_height = args.use_height
    return data_args


def get_prompt_templates(prompt_key: str) -> List[str]:
    templates_path = os.path.join(ULIP_DIR, "data", "templates.json")
    with open(templates_path, "r", encoding="utf-8") as f:
        templates = json.load(f)
    if prompt_key not in templates:
        raise ValueError(f"Prompt key not found in templates.json: {prompt_key}")
    prompt_list = templates[prompt_key]
    if not prompt_list:
        raise ValueError(f"No templates found for key: {prompt_key}")
    return [prompt_list[0]]


def load_class_names(dataset_name: str) -> List[str]:
    labels_path = os.path.join(ULIP_DIR, "data", "labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if dataset_name not in labels:
        raise ValueError(f"labels.json missing dataset: {dataset_name}")
    return labels[dataset_name]


def build_clients(
    ulip_models_map: Dict[str, torch.nn.Module],
    loaders: Sequence[DataLoader],
    args: argparse.Namespace,
    tokenizer: SimpleTokenizer,
    devices: Sequence[torch.device],
    prompt_templates: Sequence[str],
    label_names: Optional[Sequence[str]] = None,
) -> List[FedULIPClient]:
    clients = []
    share_inversion = args.share_inversion and not args.disable_inversion
    for idx, loader in enumerate(loaders):
        device = devices[idx % len(devices)]
        ulip_model = ulip_models_map[str(device)]
        client = FedULIPClient.from_ulip_model(
            ulip_model,
            adapter_ratio=args.adapter_ratio,
            saca_heads=args.saca_heads,
            inversion_hidden_dim=args.inversion_hidden_dim,
            prompt_chunk_size=args.prompt_chunk_size,
            eval_chunk_size=args.eval_chunk_size,
            prompt_templates=prompt_templates,
            tokenizer=tokenizer,
            train_loader=loader,
            label_names=label_names,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            supcon_weight=args.supcon_weight,
            supcon_mode=args.supcon_mode,
            triplet_weight=args.triplet_weight,
            triplet_margin=args.triplet_margin,
            device=device,
            use_inversion=not args.disable_inversion,
            share_inversion=share_inversion,
            use_saca=not args.disable_saca,
        )
        client.client_id = idx
        client.device_tag = str(device)
        clients.append(client)
    return clients


def evaluate(
    client: FedULIPClient,
    loader: DataLoader,
    candidate_classes: Sequence[str],
) -> Tuple[int, int]:
    if len(loader.dataset) == 0:
        return 0, 0
    device = next(client.adapter.parameters()).device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    client.eval()
    class_text_feats = None
    if not (client.use_inversion and client.pseudo_token_mode == "instance"):
        class_text_feats = client.get_class_text_features_eval(candidate_classes).to(device)
        client.set_eval_class_text_features(candidate_classes, class_text_feats)

    correct = 0
    total = 0
    prev_mode = client.pseudo_token_mode
    client.pseudo_token_mode = "instance"
    for batch in loader:
        points, labels, _ = batch
        points = points.to(device)
        labels = labels.to(device).long()
        preds = client.zero_shot_inference(points, candidate_classes)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    client.pseudo_token_mode = prev_mode
    return correct, total


def format_acc(correct: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{correct / total * 100:.2f}%"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_name = args.dataset
    dataset_spec = DATASET_SPECS[dataset_name]
    domains = dataset_spec["domains"]

    devices = resolve_devices(args.devices)
    max_concurrent = args.max_concurrent_clients or len(devices)
    tokenizer = SimpleTokenizer()
    log = setup_logging(args.output_dir, args.run_name)
    share_inversion = args.share_inversion and not args.disable_inversion
    use_global_eval = share_inversion or args.disable_inversion

    log("Parameters:")
    for key, value in sorted(vars(args).items()):
        log(f"  {key}: {value}")
    log(f"Domains: {domains}")
    if args.share_inversion and args.disable_inversion:
        log("Warning: --share-inversion ignored because --disable-inversion is set.")
    if use_global_eval and args.eval_max_samples:
        log("Note: --eval-max-samples is ignored when global evaluation is enabled.")

    train_spec = dataset_spec["train"]
    test_spec = dataset_spec["test"]
    data_args = build_dataset_args(args, train_spec, test_spec)
    train_base = get_dataset(None, tokenizer, data_args, "train")
    test_base = get_dataset(None, tokenizer, data_args, "val")

    train_pool = list(range(len(train_base)))
    if args.train_max_samples and args.train_max_samples < len(train_pool):
        rng = np.random.default_rng(args.seed)
        train_pool = rng.choice(train_pool, size=args.train_max_samples, replace=False).tolist()

    test_pool = list(range(len(test_base)))
    if not use_global_eval and args.eval_max_samples and args.eval_max_samples < len(test_pool):
        rng = np.random.default_rng(args.seed + 100)
        test_pool = rng.choice(test_pool, size=args.eval_max_samples, replace=False).tolist()

    train_domain_indices = split_indices_evenly(train_pool, domains, args.seed)
    test_domain_indices = split_indices_evenly(test_pool, domains, args.seed + 1)

    class_names = load_class_names(dataset_name)
    train_domain_datasets = {}
    for domain in domains:
        indices = train_domain_indices[domain]
        train_domain_datasets[domain] = DomainDataset(
            train_base,
            indices,
            domain,
            target_npoints=args.npoints,
            sparse_npoints=args.sparse_npoints,
            outlier_ratio=args.outlier_ratio,
            drop_ratio=args.drop_ratio,
            partial_keep_ratio=args.partial_keep_ratio,
            seed=args.seed + 1000,
            label_names=class_names,
        )

    test_domain_datasets = {}
    for domain in domains:
        indices = test_domain_indices[domain]
        test_domain_datasets[domain] = DomainDataset(
            test_base,
            indices,
            domain,
            target_npoints=args.npoints,
            sparse_npoints=args.sparse_npoints,
            outlier_ratio=args.outlier_ratio,
            drop_ratio=args.drop_ratio,
            partial_keep_ratio=args.partial_keep_ratio,
            seed=args.seed + 2000,
            label_names=class_names,
        )
    prompt_templates = get_prompt_templates(args.prompt)

    fold_indices = list(range(len(domains)))
    if args.fold >= 0:
        if args.fold not in fold_indices:
            raise ValueError(f"Invalid fold {args.fold} for domains {domains}")
        fold_indices = [args.fold]

    for fold_idx in fold_indices:
        held_out = domains[fold_idx]
        train_domains = [d for d in domains if d != held_out]
        log(f"Fold {fold_idx}: train={train_domains} test={held_out}")

        client_domains = assign_clients_to_domains(args.num_clients, train_domains)
        domain_to_clients: Dict[str, List[int]] = {d: [] for d in train_domains}
        for client_id, dom in enumerate(client_domains):
            domain_to_clients[dom].append(client_id)

        client_indices = [[] for _ in range(args.num_clients)]
        client_domain_dataset = [None] * args.num_clients
        for domain in train_domains:
            domain_dataset = train_domain_datasets[domain]
            domain_clients = domain_to_clients[domain]
            if not domain_clients:
                continue
            local_indices = list(range(len(domain_dataset)))
            partitions = split_indices_randomly(
                local_indices,
                num_clients=len(domain_clients),
                seed=args.seed + fold_idx * 10 + ord(domain),
            )
            for client_idx, client_id in enumerate(domain_clients):
                subset = partitions[client_idx]
                if not subset:
                    subset = [random.choice(local_indices)]
                client_indices[client_id] = subset
                client_domain_dataset[client_id] = domain_dataset

        client_loaders = []
        for client_id, idxs in enumerate(client_indices):
            dataset = client_domain_dataset[client_id]
            if dataset is None:
                raise ValueError(f"Client {client_id} has no assigned domain.")
            subset = Subset(dataset, idxs)
            if args.pk_sampler:
                labels = labels_for_indices(dataset, idxs)
                batch_sampler = PKBatchSampler(
                    labels,
                    num_classes=args.pk_classes,
                    num_samples=args.pk_samples,
                    drop_last=True,
                    seed=args.seed,
                )
                if args.batch_size != args.pk_classes * args.pk_samples:
                    log(
                        "Warning: batch-size does not match pk-classes * pk-samples; "
                        "batch-size will be ignored for P×K sampling."
                    )
                loader = DataLoader(
                    subset,
                    batch_sampler=batch_sampler,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=customized_collate_fn,
                )
            else:
                loader = DataLoader(
                    subset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=customized_collate_fn,
                )
            client_loaders.append(loader)

        ulip_models_map = build_ulip_models(args, devices)
        clients = build_clients(
            ulip_models_map,
            client_loaders,
            args,
            tokenizer,
            devices,
            prompt_templates,
            label_names=class_names,
        )

        if args.num_clients == 1:
            device_ids = [device.index for device in devices if device.type == "cuda" and device.index is not None]
            enable_dp = (
                (len(device_ids) > 1 and not args.disable_single_client_data_parallel)
                or args.single_client_data_parallel
            )
            if enable_dp and len(device_ids) > 1:
                clients[0].enable_data_parallel(device_ids)
                clients[0].device_tag = f"{clients[0].device_tag} (DP: {device_ids})"
                log(f"Single-client DataParallel enabled on devices: {device_ids}")

        server = FedULIPServer(
            adapter=copy.deepcopy(clients[0].adapter),
            saca=copy.deepcopy(clients[0].saca),
            logit_scale=copy.deepcopy(clients[0].logit_scale),
            inversion_net=copy.deepcopy(clients[0].inversion_net) if share_inversion else None,
        )

        client_sizes = [len(idxs) for idxs in client_indices]
        log(f"Clients: {len(clients)} | Samples per client: {client_sizes}")

        eval_dataset = test_domain_datasets[held_out]
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=customized_collate_fn,
        )

        global_state = server.get_global_state()
        if use_global_eval:
            eval_client = clients[min(args.eval_client_id, len(clients) - 1)]
            eval_client.load_global_state(global_state)
            log(
                f"Fold {fold_idx} Round -1/{args.rounds} global eval start "
                f"(device={eval_client.device_tag})"
            )
            correct, total = evaluate(eval_client, eval_loader, class_names)
            log(
                f"Fold {fold_idx} Round -1/{args.rounds} global acc: "
                f"{format_acc(correct, total)} (n={total})"
            )
            log(
                f"Fold {fold_idx} Round -1/{args.rounds} global eval end "
                f"(device={eval_client.device_tag})"
            )
        else:
            total_correct = 0
            total_seen = 0
            for client_id, client in enumerate(clients):
                client.load_global_state(global_state)
                log(
                    f"Fold {fold_idx} Round -1/{args.rounds} eval client {client_id} start "
                    f"(device={client.device_tag})"
                )
                correct, total = evaluate(client, eval_loader, class_names)
                total_correct += correct
                total_seen += total
                log(
                    f"Fold {fold_idx} Round -1/{args.rounds} client {client_id} acc: "
                    f"{format_acc(correct, total)} (n={total})"
                )
                log(
                    f"Fold {fold_idx} Round -1/{args.rounds} eval client {client_id} end "
                    f"(device={client.device_tag})"
                )
            log(
                f"Fold {fold_idx} Round -1/{args.rounds} summary | "
                f"avg_acc={format_acc(total_correct, total_seen)} (n={total_seen})"
            )

        for round_idx in range(args.rounds):
            round_start = time.time()
            global_state = server.get_global_state()
            client_updates = [None] * len(clients)
            if max_concurrent <= 1:
                for client_id, client in enumerate(clients):
                    log(
                        f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} "
                        f"client {client_id} start (device={client.device_tag})"
                    )
                    client_updates[client_id] = client.local_train(global_state, local_epochs=args.local_epochs)
                    stats = client_updates[client_id].get("stats", {})
                    if stats:
                        log(
                            f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} client {client_id} stats | "
                            f"loss={stats.get('avg_loss', 0):.4f} | "
                            f"ce={stats.get('avg_ce', 0):.4f} | "
                            f"supcon={stats.get('avg_supcon', 0):.4f} | "
                            f"triplet={stats.get('avg_triplet', 0):.4f} | "
                            f"acc={stats.get('avg_acc', 0) * 100:.2f}% | "
                            f"s*={stats.get('avg_s_star_norm', 0):.3f} | "
                            f"raw={stats.get('avg_raw_norm', 0):.3f} | "
                            f"pc={stats.get('avg_pc_norm', 0):.3f} | "
                            f"logit_scale={stats.get('logit_scale', 0):.3f}"
                        )
                    log(
                        f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} "
                        f"client {client_id} end (device={client.device_tag})"
                    )
            else:
                device_queues: Dict[torch.device, List[tuple]] = {}
                for client_id, client in enumerate(clients):
                    device = next(client.adapter.parameters()).device
                    device_queues.setdefault(device, []).append((client_id, client))

                while any(device_queues.values()):
                    futures = []
                    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                        for queue in device_queues.values():
                            if not queue:
                                continue
                            client_id, client = queue.pop(0)
                            log(
                                f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} client {client_id} start "
                                f"(device={client.device_tag})"
                            )
                            futures.append((client_id, executor.submit(
                                client.local_train, global_state, args.local_epochs)))
                        for client_id, future in futures:
                            client_updates[client_id] = future.result()
                            stats = client_updates[client_id].get("stats", {})
                            if stats:
                                log(
                                    f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} client {client_id} stats | "
                                    f"loss={stats.get('avg_loss', 0):.4f} | "
                                    f"ce={stats.get('avg_ce', 0):.4f} | "
                                    f"supcon={stats.get('avg_supcon', 0):.4f} | "
                                    f"triplet={stats.get('avg_triplet', 0):.4f} | "
                                    f"acc={stats.get('avg_acc', 0) * 100:.2f}% | "
                                    f"s*={stats.get('avg_s_star_norm', 0):.3f} | "
                                    f"raw={stats.get('avg_raw_norm', 0):.3f} | "
                                    f"pc={stats.get('avg_pc_norm', 0):.3f} | "
                                    f"logit_scale={stats.get('logit_scale', 0):.3f}"
                                )
                            log(
                                f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} client {client_id} end "
                                f"(device={clients[client_id].device_tag})"
                            )

            server.aggregate(client_updates, weights=client_sizes)
            train_time = time.time() - round_start
            global_state = server.get_global_state()
            log(f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} aggregation done")

            if use_global_eval:
                eval_client = clients[min(args.eval_client_id, len(clients) - 1)]
                eval_client.load_global_state(global_state)
                log(
                    f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} global eval start "
                    f"(device={eval_client.device_tag})"
                )
                correct, total = evaluate(eval_client, eval_loader, class_names)
                log(
                    f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} global acc: "
                    f"{format_acc(correct, total)} (n={total})"
                )
                log(
                    f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} global eval end "
                    f"(device={eval_client.device_tag})"
                )
                log(
                    f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} summary | "
                    f"train_time={train_time:.1f}s | "
                    f"acc={format_acc(correct, total)} (n={total})"
                )
            else:
                total_correct = 0
                total_seen = 0
                for client_id, client in enumerate(clients):
                    client.load_global_state(global_state)
                    log(
                        f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} eval client {client_id} start "
                        f"(device={client.device_tag})"
                    )
                    correct, total = evaluate(client, eval_loader, class_names)
                    total_correct += correct
                    total_seen += total
                    log(
                        f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} client {client_id} acc: "
                        f"{format_acc(correct, total)} (n={total})"
                    )
                    log(
                        f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} eval client {client_id} end "
                        f"(device={client.device_tag})"
                    )

                log(
                    f"Fold {fold_idx} Round {round_idx + 1}/{args.rounds} summary | "
                    f"train_time={train_time:.1f}s | "
                    f"avg_acc={format_acc(total_correct, total_seen)} (n={total_seen})"
                )

        eval_client = clients[min(args.eval_client_id, len(clients) - 1)]
        eval_client.load_global_state(server.get_global_state())
        correct, total = evaluate(eval_client, eval_loader, class_names)
        if use_global_eval:
            log(
                f"Fold {fold_idx} Final eval global: "
                f"{format_acc(correct, total)} (n={total})"
            )
        else:
            log(
                f"Fold {fold_idx} Final eval client {args.eval_client_id}: "
                f"{format_acc(correct, total)} (n={total})"
            )


if __name__ == "__main__":
    main()
