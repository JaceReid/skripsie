# make_10_shards_for_cv.py
import os, json, h5py, argparse, numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

def read_keys_and_labels(h5_path):
    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())
    labels_text = [k.split("_")[0] for k in keys]        # adjust if your label rule differs
    classes = sorted(set(labels_text))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[t] for t in labels_text], dtype=np.int64)
    return keys, y, classes

def write_subset_h5(src_h5, keys_subset, out_path, compress=True):
    comp = dict(compression="gzip", compression_opts=4, shuffle=True) if compress else {}
    with h5py.File(src_h5, "r") as src, h5py.File(out_path, "w") as dst:
        for k in tqdm(keys_subset, desc=f"Writing {os.path.basename(out_path)}"):
            dst.create_dataset(k, data=src[k][()], dtype=src[k].dtype, **comp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_h5",default="./Datasets/FD_5.1.h5" , help="Master HDF5, e.g. ./Datasets/FD_3.0.h5")
    ap.add_argument("--out_dir", default="./Datasets/folds_10_5.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_compress", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    keys, y, classes = read_keys_and_labels(args.in_h5)
    n = len(keys)
    print(f"Found {n} items across {len(classes)} classes: {classes}")

    # Build 10 **disjoint** parts (stratified)
    skf = StratifiedKFold(n_splits=11, shuffle=True, random_state=args.seed)
    idx2key = np.array(keys)

    # Collect val indices per fold; each is a disjoint 10% “shard”
    shard_key_lists = []
    for fold_id, (_, val_idx) in enumerate(skf.split(np.zeros(n), y)):
        shard_keys = idx2key[val_idx].tolist()
        shard_key_lists.append(shard_keys)

    # Write each shard once
    shard_paths = []
    for i, shard_keys in enumerate(shard_key_lists):
        shard_path = os.path.join(args.out_dir, f"shard{i:02d}.h5")
        print(f"\n[shard{i:02d}] {len(shard_keys)} samples -> {shard_path}")
        write_subset_h5(args.in_h5, shard_keys, shard_path, compress=not args.no_compress)
        shard_paths.append(os.path.abspath(shard_path))

    # Create a folds mapping: for fold k, val = shard k; train = all other shards
    folds = []
    for k in range(10):
        train_shards = [p for i, p in enumerate(shard_paths) if i != k]
        val_shards   = [shard_paths[k]]
        folds.append({
            "fold": k+1,
            "train_shards": train_shards,
            "val_shards": val_shards
        })

    meta = {
        "in_h5": os.path.abspath(args.in_h5),
        "classes": classes,
        "seed": args.seed,
        "shards": shard_paths,
        "folds": folds
    }
    meta_path = os.path.join(args.out_dir, "folds_shards.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote mapping to {meta_path}")
    print("Done.")
    
if __name__ == "__main__":
    main()
