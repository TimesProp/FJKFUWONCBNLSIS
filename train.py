"""Standalone Type7 training, adapted from TimesNetDHAblationMulti.py."""
import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from model import Model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'datasets/mixedhumid_sfd.csv'
HIER_PATH = ROOT / 'datasets/mixedhumid_sfd_hier.csv'
OUTPUT_DIR = ROOT / 'outputs'
RUN_TIMES = 3
MODEL_NAME = 'type7'
LOSS_NAME = 'huber'
DATA_RATIO = 1
SEQ_LEN = 24
PRED_LEN = 24
TIME_HIERARCHY = [1, 2, 3]
BASE_INTERVAL_MIN = 60
DH_STEPS = 1
RUN_TAG = ''
DH_HIDDEN = 16
MAX_RESIDUAL_GATE = 0.5
COHERENCE_LOSS_WEIGHT = 0.01
BU_LOSS_WEIGHT = 0.01
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 3
D_MODEL = 16
D_FF = 64
E_LAYERS = 2
TOP_K = 5
NUM_KERNELS = 6
DROPOUT = 0.1
SEED = 42
NUM_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE_TEST = False

class TimeSeriesDataset(Dataset):

    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len:index + self.seq_len + self.pred_len]
        return (x, y)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def aggregate(x, factor):
    if factor == 1:
        return x
    T = x.shape[-2] // factor * factor
    if x.ndim == 2:
        return x[:T].reshape(-1, factor, x.shape[-1]).sum(axis=1)
    return x[:, :T].reshape(x.shape[0], -1, factor, x.shape[-1]).sum(axis=2)

def build_multiscale(raw, time_hierarchy):
    levels = [raw]
    current = raw
    for ratio in time_hierarchy[1:]:
        current = aggregate(current, ratio)
        levels.append(current)
    return levels

def build_scale_factors(time_hierarchy):
    factors = []
    cumulative = 1
    for ratio in time_hierarchy:
        cumulative *= ratio
        factors.append(cumulative)
    return factors

def format_interval(minutes):
    if minutes < 60:
        return f'{minutes}min'
    if minutes % 60 == 0:
        hours = minutes // 60
        return f'{hours}hour' if hours == 1 else f'{hours}hours'
    return f'{minutes}min'

def get_stats(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-06] = 1.0
    return (mean, std)

def load_hierarchy(path, node_names):
    df = pd.read_csv(path, index_col=0)
    missing = [n for n in node_names if n not in df.index or n not in df.columns]
    if missing:
        raise ValueError(f'Nodes missing in hierarchy: {missing}')
    df = df.loc[node_names, node_names]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    adj = df.values.astype(np.float32)
    children = {node: [] for node in node_names}
    parents = {node: [] for node in node_names}
    for i, parent in enumerate(node_names):
        for j, child in enumerate(node_names):
            if adj[i, j] > 0:
                children[parent].append(child)
                parents[child].append(parent)
    roots = [node for node in node_names if len(parents[node]) == 0]
    if len(roots) == 0:
        raise ValueError('No root node found in hierarchy.')
    node_levels = np.full(len(node_names), -1, dtype=int)
    current = roots
    level = 0
    while current:
        next_nodes = []
        for node in current:
            idx = node_names.index(node)
            if node_levels[idx] == -1 or level < node_levels[idx]:
                node_levels[idx] = level
            next_nodes.extend(children[node])
        current = list(dict.fromkeys(next_nodes))
        level += 1
    if np.any(node_levels < 0):
        missing_nodes = [node_names[i] for i in range(len(node_names)) if node_levels[i] < 0]
        raise ValueError(f'Hierarchy contains unreachable nodes: {missing_nodes}')
    return (adj, children, parents, roots, node_levels)

def bottom_up_reconcile(pred, node_names, children, node_levels):
    result = pred.copy()
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    max_level = node_levels.max()
    for level in range(max_level - 1, -1, -1):
        for i, node in enumerate(node_names):
            if node_levels[i] == level and len(children[node]) > 0:
                child_idx = [name_to_idx[c] for c in children[node]]
                result[:, :, i] = result[:, :, child_idx].sum(axis=2)
    return result

def top_down_reconcile(pred, node_names, children, node_levels):
    result = pred.copy()
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    max_level = node_levels.max()
    eps = 1e-08
    for level in range(max_level):
        for i, parent in enumerate(node_names):
            if node_levels[i] != level or len(children[parent]) == 0:
                continue
            child_idx = [name_to_idx[c] for c in children[parent]]
            original_children = pred[:, :, child_idx]
            child_sum = original_children.sum(axis=2, keepdims=True)
            proportions = original_children / np.where(np.abs(child_sum) < eps, 1.0, child_sum)
            zero_mask = np.abs(child_sum) < eps
            proportions = np.where(zero_mask, 1.0 / len(child_idx), proportions)
            result[:, :, child_idx] = result[:, :, i:i + 1] * proportions
    return result

def load_data():
    df = pd.read_csv(DATA_PATH)
    if not 0 < DATA_RATIO <= 1:
        raise ValueError('DATA_RATIO must be in (0, 1].')
    original_len = len(df)
    use_len = int(original_len * DATA_RATIO)
    df = df.iloc[:use_len].reset_index(drop=True)
    print('Original CSV shape:', (original_len, df.shape[1]))
    print('Used CSV shape:', df.shape)
    print(f'DATA_RATIO: {DATA_RATIO:.2f}')
    print('Time column:', df.columns[0])
    node_names = df.columns[1:].tolist()
    data = df.iloc[:, 1:].values.astype(np.float32)
    print('Number of variables:', len(node_names))
    print('Time series length:', len(data))
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    if not np.isfinite(data).all():
        raise ValueError("Dataset contains NaN or infinite values.")
    if train_end < SEQ_LEN + PRED_LEN or val_end - train_end < PRED_LEN or n - val_end < PRED_LEN:
        raise ValueError("Dataset splits are too short for the selected sequence/horizon.")
    scaler = StandardScaler()
    scaler.fit(data[:train_end])
    data_scaled = scaler.transform(data)
    train_data = data_scaled[:train_end]
    val_data = data_scaled[train_end - SEQ_LEN:val_end]
    test_data = data_scaled[val_end - SEQ_LEN:]
    print('Train raw range: 0 -', train_end)
    print('Val raw range:', train_end, '-', val_end)
    print('Test raw range:', val_end, '-', n)
    train_dataset = TimeSeriesDataset(train_data, SEQ_LEN, PRED_LEN)
    val_dataset = TimeSeriesDataset(val_data, SEQ_LEN, PRED_LEN)
    test_dataset = TimeSeriesDataset(test_data, SEQ_LEN, PRED_LEN)
    if SMOKE_TEST:
        train_dataset = Subset(train_dataset, range(min(BATCH_SIZE, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(BATCH_SIZE, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(BATCH_SIZE, len(test_dataset))))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    train_levels = build_multiscale(data[:train_end], TIME_HIERARCHY)
    scale_stats = [get_stats(level) for level in train_levels]
    return (train_loader, val_loader, test_loader, scaler, scale_stats, node_names)

def build_model(num_nodes, adj, scaler, scale_stats):
    configs = SimpleNamespace()
    configs.task_name = 'long_term_forecast'
    configs.seq_len = SEQ_LEN
    configs.label_len = 0
    configs.pred_len = PRED_LEN
    configs.enc_in = num_nodes
    configs.c_out = num_nodes
    configs.d_model = D_MODEL
    configs.d_ff = D_FF
    configs.e_layers = E_LAYERS
    configs.top_k = TOP_K
    configs.num_kernels = NUM_KERNELS
    configs.embed = 'fixed'
    configs.freq = 'h'
    configs.dropout = DROPOUT
    configs.enable_dh = True
    configs.adj = torch.as_tensor(adj, dtype=torch.float32)
    configs.time_factors = build_scale_factors(TIME_HIERARCHY)
    configs.dh_hidden = DH_HIDDEN
    configs.dh_steps = DH_STEPS
    configs.max_residual_gate = MAX_RESIDUAL_GATE
    configs.coherence_loss_weight = COHERENCE_LOSS_WEIGHT
    configs.bu_loss_weight = BU_LOSS_WEIGHT
    configs.scaler_mean = scaler.mean_.astype(np.float32)
    configs.scaler_scale = scaler.scale_.astype(np.float32)
    configs.time_scaler_scales = np.concatenate([std for _, std in scale_stats], axis=0).astype(np.float32)
    return Model(configs)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x, None, None, None)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            total_num += x.size(0)
    return total_loss / total_num

def train(model, train_loader, val_loader, model_version):
    criterion = nn.HuberLoss() if LOSS_NAME == 'huber' else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float('inf')
    patience_count = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f'{model_version}_best.pth'
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_main_loss = 0
        total_num = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x, None, None, None)
            main_loss = criterion(pred, y)
            loss = model.compute_loss(pred, y, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_main_loss += main_loss.item() * x.size(0)
            total_num += x.size(0)
        train_loss = total_loss / total_num
        train_main_loss = total_main_loss / total_num
        val_loss = validate(model, val_loader, criterion)
        print(f'Epoch {epoch:03d} | Train total={train_loss:.6f} | Train {LOSS_NAME}={train_main_loss:.6f} | Val {LOSS_NAME}={val_loss:.6f} | Coherence={model.last_coherence_loss.item():.6f} | BU={model.last_bu_loss.item():.6f}')
        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print('Best model saved.')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print('Early stopping.')
                break
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    return model

def format_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    return np.char.add(np.char.add(np.char.mod('%.6f', mean), '±'), np.char.mod('%.6f', std))

def print_summary(results):
    print(f'\nCompleted runs: {len(results)} | Sample std (ddof=1); single run std=0')
    print('========== TEST ==========')
    for metric in results[0]['metrics']:
        value = format_mean_std([r['metrics'][metric] for r in results])
        print(f'{metric}: {value.item()}')
    tables = {metric: np.stack([r['tables'][metric] for r in results]) for metric in results[0]['tables']}
    scales = results[0]['scale_names']
    levels = results[0]['data_level_names']
    for metric, values in tables.items():
        print(f'\n========== {metric} ==========')
        print('\t'.join(['Time'] + levels))
        for name, row in zip(scales, format_mean_std(values)):
            print('\t'.join([name] + row.tolist()))
    for title, names, axis in [('AVERAGE ACROSS TIME LEVELS', levels, 1), ('AVERAGE ACROSS DATA LEVELS', scales, 2)]:
        print(f'\n========== {title} ==========')
        print('\t'.join(['Metric'] + names))
        for metric, values in tables.items():
            row = format_mean_std(values.mean(axis=axis))
            print('\t'.join([metric] + row.tolist()))

def test(model, test_loader, scaler, scale_stats, node_names):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(DEVICE), None, None, None).cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse_norm = np.mean((preds - trues) ** 2)
    mae_norm = np.mean(np.abs(preds - trues))
    B, T, N = preds.shape
    preds_real = scaler.inverse_transform(preds.reshape(-1, N)).reshape(B, T, N)
    trues_real = scaler.inverse_transform(trues.reshape(-1, N)).reshape(B, T, N)
    mse = np.mean((preds_real - trues_real) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_real - trues_real))
    adj, children, parents, roots, node_levels = load_hierarchy(HIER_PATH, node_names)
    num_data_levels = node_levels.max() + 1
    data_level_names = [f'L{i}' for i in range(num_data_levels)]
    print()
    pred_bu_real = bottom_up_reconcile(preds_real, node_names, children, node_levels)
    pred_td_real = top_down_reconcile(preds_real, node_names, children, node_levels)
    pred_bu_levels = build_multiscale(pred_bu_real, TIME_HIERARCHY)
    pred_td_levels = build_multiscale(pred_td_real, TIME_HIERARCHY)
    true_levels = build_multiscale(trues_real, TIME_HIERARCHY)
    scale_factors = build_scale_factors(TIME_HIERARCHY)
    scale_names = [format_interval(BASE_INTERVAL_MIN * factor) for factor in scale_factors]
    bu_mse_table = np.zeros((len(scale_names), num_data_levels))
    td_mse_table = np.zeros((len(scale_names), num_data_levels))
    bu_mae_table = np.zeros((len(scale_names), num_data_levels))
    td_mae_table = np.zeros((len(scale_names), num_data_levels))
    for scale_idx in range(len(scale_names)):
        mean, std = scale_stats[scale_idx]
        pred_bu_norm = (pred_bu_levels[scale_idx] - mean) / std
        pred_td_norm = (pred_td_levels[scale_idx] - mean) / std
        true_norm = (true_levels[scale_idx] - mean) / std
        for level in range(num_data_levels):
            mask = node_levels == level
            bu_error = pred_bu_norm[:, :, mask] - true_norm[:, :, mask]
            td_error = pred_td_norm[:, :, mask] - true_norm[:, :, mask]
            bu_mse_table[scale_idx, level] = np.mean(bu_error ** 2)
            td_mse_table[scale_idx, level] = np.mean(td_error ** 2)
            bu_mae_table[scale_idx, level] = np.mean(np.abs(bu_error))
            td_mae_table[scale_idx, level] = np.mean(np.abs(td_error))
    return {'metrics': {'Normalized MSE': mse_norm, 'Normalized MAE': mae_norm, 'Real-scale MSE': mse, 'Real-scale RMSE': rmse, 'Real-scale MAE': mae}, 'tables': {'BU MSE': bu_mse_table, 'TD MSE': td_mse_table, 'BU MAE': bu_mae_table, 'TD MAE': td_mae_table}, 'scale_names': scale_names, 'data_level_names': data_level_names}

def main():
    if isinstance(RUN_TIMES, bool) or not isinstance(RUN_TIMES, int) or RUN_TIMES < 1:
        raise ValueError('RUN_TIMES must be a positive integer.')
    print('Device:', DEVICE)
    print('Selected model:', MODEL_NAME, '| Main loss:', LOSS_NAME)
    scale_factors = build_scale_factors(TIME_HIERARCHY)
    scale_names = [format_interval(BASE_INTERVAL_MIN * factor) for factor in scale_factors]
    print('TIME_HIERARCHY:', TIME_HIERARCHY)
    print('CUMULATIVE FACTORS:', scale_factors)
    print('TIME SCALES:', scale_names)
    if SEQ_LEN % scale_factors[-1] != 0:
        raise ValueError(f'SEQ_LEN={SEQ_LEN} must be divisible by {scale_factors[-1]}.')
    if PRED_LEN % scale_factors[-1] != 0:
        raise ValueError(f'PRED_LEN={PRED_LEN} must be divisible by {scale_factors[-1]}.')
    results = []
    for run_index in range(RUN_TIMES):
        run_seed = SEED + run_index
        seed_everything(run_seed)
        model_version = f'ablation_multi_{Path(DATA_PATH).stem}_{MODEL_NAME}_seed{run_seed}'
        if RUN_TAG:
            model_version += f'_{RUN_TAG}'
        print(f'\n========== RUN {run_index + 1}/{RUN_TIMES} | SEED {run_seed} ==========')
        train_loader, val_loader, test_loader, scaler, scale_stats, node_names = load_data()
        adj, _, _, _, _ = load_hierarchy(HIER_PATH, node_names)
        model = build_model(len(node_names), adj, scaler, scale_stats).to(DEVICE)
        print('Building:', model_version)
        print('Total parameters:', sum((p.numel() for p in model.parameters())))
        print('Trainable parameters:', sum((p.numel() for p in model.parameters() if p.requires_grad)))
        model = train(model, train_loader, val_loader, model_version)
        results.append(test(model, test_loader, scaler, scale_stats, node_names))
        del model, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in globals().items() if key.isupper()}, "runs": results}, handle, indent=2, default=lambda value: value.tolist())
    print_summary(results)
    print('Evaluation complete.')

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='mixedhumid_sfd', help='CSV basename in datasets/')
    parser.add_argument('--runs', type=int, default=RUN_TIMES)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seq', type=int, default=SEQ_LEN)
    parser.add_argument('--pred-len', type=int, default=PRED_LEN)
    parser.add_argument('--dh-steps', '--dh_steps', type=int, default=DH_STEPS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=DEVICE.type)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--smoke-test', action='store_true', help='One batch per split; one epoch/run; does not reproduce paper metrics')
    args = parser.parse_args()
    for name in ['runs', 'epochs', 'batch_size', 'seq', 'pred_len', 'dh_steps']:
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.seq + args.pred_len < 2 * TOP_K:
        parser.error('Sequence plus forecast length is too small for TOP_K')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu')
    if not (ROOT / 'datasets' / f'{args.dataset}.csv').is_file():
        parser.error(f'Unknown dataset: {args.dataset}')
    return args


if __name__ == '__main__':
    args = parse_args()
    DATA_PATH = ROOT / 'datasets' / f'{args.dataset}.csv'
    HIER_PATH = ROOT / 'datasets' / f'{args.dataset}_hier.csv'
    RUN_TIMES, EPOCHS = args.runs, args.epochs
    BATCH_SIZE = args.batch_size
    SEQ_LEN, PRED_LEN = args.seq, args.pred_len
    DH_STEPS, SEED = args.dh_steps, args.seed
    DEVICE = torch.device(args.device)
    OUTPUT_DIR = args.output_dir.resolve()
    SMOKE_TEST = args.smoke_test
    if SMOKE_TEST:
        RUN_TIMES = EPOCHS = 1
        BATCH_SIZE = min(BATCH_SIZE, 2)
        OUTPUT_DIR = OUTPUT_DIR / 'smoke'
        print('SMOKE TEST: one batch per split; metrics are not paper results.')
    main()
