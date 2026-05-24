#!/usr/bin/env python3

import argparse

import numpy as np

import _pb_gpg


def make_data(rows: int = 128):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    y = (x[:, 0] * x[:, 1] + 0.5 * x[:, 2]).astype(np.float32)
    return x, y


def run_once(backend: str, gpu_batch_size: int):
    x, y = make_data()
    options = (
        f"-backend {backend} "
        "-ff mse "
        "-fset +,-,*,aq "
        "-disable_ims "
        "-g 1 "
        "-pop 32 "
        "-d 3 "
        "-cmp 0 "
        "-feat_sel -1 "
        f"-gpu_batch_size {gpu_batch_size} "
        "-random_state 0"
    )
    models = _pb_gpg.evolve(options, x, y)
    if len(models) == 0:
        raise AssertionError("pybind evolve returned no models")
    return [str(model) for model in models]


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the pybind GPG interface.")
    parser.add_argument("--backend", default="cpu_original")
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    if args.backend != "cpu_original" and not _pb_gpg.cuda_enabled():
        raise SystemExit(f"backend {args.backend!r} requires a CUDA-enabled _pb_gpg build")

    print("cuda_enabled=", _pb_gpg.cuda_enabled())
    print("available_backends=", list(_pb_gpg.available_backends()))

    for i in range(args.repeat):
        models = run_once(args.backend, args.gpu_batch_size)
        print(f"run {i}: {len(models)} models; first={models[0]}")


if __name__ == "__main__":
    main()
