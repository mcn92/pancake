#!/usr/bin/env python3
"""
Convert fvecs/ivecs files into an HDF5 file matching the ann-benchmarks layout.

Output HDF5 contains:
  /train      float32  [n_train, dim]     base vectors
  /test       float32  [n_test, dim]      query vectors
  /neighbors  int32    [n_test, k]        ground-truth top-k neighbor IDs
  /distances  float32  [n_test, k]        OPTIONAL — only if distances file provided
  (attributes: distance = 'angular' or 'euclidean')

Usage:
  python3 fvecs_to_hdf5.py <base.fvecs> <query.fvecs> <gt.ivecs> <out.hdf5> [options]

Options:
  --metric {angular,euclidean}   Distance metric to label the file with.
                                 Default: angular (for cosine/normalized data).
  --normalize                    L2-normalize base + query vectors. Recommended
                                 for angular data if the source vectors aren't
                                 already unit-norm.
  --distances PATH.fvecs         Also store ground-truth distances (.fvecs with
                                 float32 distance per neighbor).
  --k INT                        Truncate ground truth to top-k. Default: keep
                                 whatever's in the file.

Examples:
  # NYTimes (cosine, usually needs normalization):
  python3 fvecs_to_hdf5.py nytimes_base.fvecs nytimes_query.fvecs \\
      nytimes_groundtruth.ivecs nytimes-256-angular.hdf5 \\
      --metric angular --normalize

  # SIFT (L2, no normalization):
  python3 fvecs_to_hdf5.py sift_base.fvecs sift_query.fvecs \\
      sift_groundtruth.ivecs sift-128-euclidean.hdf5 \\
      --metric euclidean
"""
import argparse
import sys
import numpy as np
import h5py


def read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs file -> (n, dim) float32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path}: empty file")
    dim = int(raw[0])
    if dim <= 0 or dim > 100_000:
        raise ValueError(f"{path}: bogus dim={dim} (file probably not fvecs)")
    record = 1 + dim
    if raw.size % record != 0:
        raise ValueError(
            f"{path}: size {raw.size * 4} bytes not divisible by "
            f"record size {record * 4} (dim={dim})"
        )
    n = raw.size // record
    arr = raw.reshape(n, record)
    # Verify dims are consistent
    bad = np.nonzero(arr[:, 0] != dim)[0]
    if bad.size:
        raise ValueError(f"{path}: inconsistent dim at record {bad[0]}")
    # Drop the dim column and reinterpret the remaining int32s as float32s
    return arr[:, 1:].view(np.float32).copy()


def read_ivecs(path: str) -> np.ndarray:
    """Read .ivecs file -> (n, dim) int32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path}: empty file")
    dim = int(raw[0])
    if dim <= 0 or dim > 100_000:
        raise ValueError(f"{path}: bogus dim={dim}")
    record = 1 + dim
    if raw.size % record != 0:
        raise ValueError(
            f"{path}: size {raw.size * 4} bytes not divisible by "
            f"record size {record * 4} (dim={dim})"
        )
    n = raw.size // record
    arr = raw.reshape(n, record)
    bad = np.nonzero(arr[:, 0] != dim)[0]
    if bad.size:
        raise ValueError(f"{path}: inconsistent dim at record {bad[0]}")
    return arr[:, 1:].copy()


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """In-place-ish L2 normalization. Returns a new float32 array."""
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid divide-by-zero for any all-zero rows
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base",  help="Base/training vectors (.fvecs)")
    ap.add_argument("query", help="Query/test vectors (.fvecs)")
    ap.add_argument("gt",    help="Ground-truth neighbors (.ivecs)")
    ap.add_argument("out",   help="Output HDF5 file")
    ap.add_argument("--metric", choices=["angular", "euclidean"], default="angular",
                    help="Distance metric label (default: angular)")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize base and query vectors")
    ap.add_argument("--distances", default=None,
                    help="Optional: .fvecs file with ground-truth distances")
    ap.add_argument("--k", type=int, default=None,
                    help="Truncate ground-truth to top-k (default: keep all)")
    args = ap.parse_args()

    print(f"Reading {args.base} ...", flush=True)
    train = read_fvecs(args.base)
    print(f"  {train.shape[0]:,} × {train.shape[1]}d float32")

    print(f"Reading {args.query} ...", flush=True)
    test = read_fvecs(args.query)
    print(f"  {test.shape[0]:,} × {test.shape[1]}d float32")

    if train.shape[1] != test.shape[1]:
        print(f"ERROR: dim mismatch (base={train.shape[1]}, query={test.shape[1]})",
              file=sys.stderr)
        sys.exit(1)

    print(f"Reading {args.gt} ...", flush=True)
    neighbors = read_ivecs(args.gt)
    print(f"  {neighbors.shape[0]:,} × {neighbors.shape[1]}d int32")

    if neighbors.shape[0] != test.shape[0]:
        print(f"ERROR: gt rows ({neighbors.shape[0]}) != query rows ({test.shape[0]})",
              file=sys.stderr)
        sys.exit(1)

    distances = None
    if args.distances:
        print(f"Reading {args.distances} ...", flush=True)
        distances = read_fvecs(args.distances)
        print(f"  {distances.shape[0]:,} × {distances.shape[1]}d float32")
        if distances.shape != neighbors.shape:
            print(f"ERROR: distances shape {distances.shape} != "
                  f"neighbors shape {neighbors.shape}", file=sys.stderr)
            sys.exit(1)

    if args.k is not None:
        if args.k > neighbors.shape[1]:
            print(f"ERROR: requested k={args.k} but gt only has "
                  f"{neighbors.shape[1]} columns", file=sys.stderr)
            sys.exit(1)
        print(f"Truncating ground truth to top-{args.k}")
        neighbors = neighbors[:, :args.k]
        if distances is not None:
            distances = distances[:, :args.k]

    if args.normalize:
        print("L2-normalizing base + query vectors ...")
        train = l2_normalize_rows(train)
        test  = l2_normalize_rows(test)

    print(f"Writing {args.out} ...")
    with h5py.File(args.out, "w") as f:
        f.create_dataset("train",     data=train,     compression=None)
        f.create_dataset("test",      data=test,      compression=None)
        f.create_dataset("neighbors", data=neighbors, compression=None)
        if distances is not None:
            f.create_dataset("distances", data=distances, compression=None)
        f.attrs["distance"] = args.metric
        f.attrs["point_type"] = "float"

    print(f"\nDone. Summary:")
    print(f"  train:     {train.shape}")
    print(f"  test:      {test.shape}")
    print(f"  neighbors: {neighbors.shape}")
    if distances is not None:
        print(f"  distances: {distances.shape}")
    print(f"  metric:    {args.metric}")
    print(f"  output:    {args.out}")


if __name__ == "__main__":
    main()