# Hardware and inference environments

The 36 native validation runs used the host below, one run at a time. These
specifications describe the validated machine; they are not minimum requirements.

| Component | Validated configuration |
|---|---|
| GPU | 1 × NVIDIA GeForce RTX 4060 Ti, 16 GB (16,380 MiB reported by the driver) |
| NVIDIA driver | 580.178.04 |
| CPU | Intel Core i5-12600KF, 10 cores / 16 logical CPUs |
| System RAM | Approximately 32 GB; 30.65 GiB visible to Linux |
| Operating system | Ubuntu 26.04.1 LTS, x86-64 |
| Linux kernel | 7.0.0-38-generic |

## Python environments

| Package | Main environment: all models except DDLP | DDLP environment |
|---|---|---|
| Python | 3.10.19 | 3.10.20 |
| PyTorch | 2.6.0+cu124 | 2.6.0+cu126 |
| torchvision | 0.21.0+cu124 | 0.21.0+cu126 |
| CUDA runtime bundled with PyTorch | 12.4 | 12.6 |
| NumPy | 2.2.6 | 2.2.6 |
| SciPy | 1.15.3 | 1.15.2 |
| Pillow | 12.2.0 | 12.2.0 |

CUDA versions above identify the PyTorch wheels, not a separately installed
CUDA toolkit. The NVIDIA driver is shared by both environments.
[environments/validated_runtime.json](environments/validated_runtime.json)
records the inspected versions and hardware. The larger
[environments/archived_runtime.json](environments/archived_runtime.json)
is the original main-environment inventory; it includes unrelated packages and
duplicate distribution metadata and is not a clean environment lockfile.

Create and activate the main environment, then run these commands from the
repository root:

```bash
conda create -n phyevo-main python=3.10.19 pip -y
conda activate phyevo-main
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-inference.txt
```

The original DINO PCA checkpoint requires scikit-learn 1.5.2. SlotFormer imports
MoviePy's older `editor` module. The release adapters prioritize the following
local dependency directories, preserving the main environment's other packages:

```bash
python -m pip install --no-deps --target third_party/dino_foresight_python_deps scikit-learn==1.5.2
python -m pip install --no-deps --target third_party/slotformer_python_deps moviepy==1.0.3 proglog==0.1.10 pycocotools==2.0.10
```

Prepare the DDLP environment separately:

```bash
conda create -n phyevo-ddlp python=3.10.20 pip -y
conda activate phyevo-ddlp
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-inference.txt
python -m pip install scipy==1.15.2 piqa==1.3.2
```

Return to the main environment and select both interpreters:

```bash
conda activate phyevo-main
export PHYEVO_PYTHON="$(command -v python)"
export PHYEVO_DDLP_PYTHON="$(conda run -n phyevo-ddlp python -c 'import sys; print(sys.executable)')"
python scripts/run_evaluation.py --check-imports
```

A private `configs/runtime_interpreters.local.json` with a `ddlp` key is also
supported; it is ignored by Git and excluded from the release source archive.
Import checks do not load models or validate CUDA kernels. The existing
environments completed all native runs; a fresh dependency installation using
the commands above has not been tested.

## Execution and memory settings

- Single GPU (`cuda:0`), serial execution; no concurrent model evaluations.
- Two PyTorch CPU threads and launcher `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
- Seed 0 at the release entry point; native model-specific seeding is retained.
- `memory_mode=safe` where supported, with a PyTorch GPU allocator fraction of
  0.45. This fraction does not limit host RAM or memory allocated outside PyTorch.
- Before each run, the launcher requires at least 12 GiB available RAM
  (22 GiB for V-JEPA 2 ViT-G), GPU memory use at most 1,024 MiB, and GPU
  utilization at most 10%. `--wait-for-resources` waits for these conditions.
- The validation service additionally used Linux memory limits: `MemoryHigh=22G`,
  `MemoryMax=24G`, `MemorySwapMax=2G`, and `CPUQuota=200%`. These are service-level
  safeguards, not settings imposed by `reproduce.sh` itself.

After restoring the original data/checkpoints and fetching the pinned OCVP code
as described in the README, run:

```bash
bash reproduce.sh --mode full --wait-for-resources
```

On a Linux host with a user systemd manager, the optional matching memory limits
can be applied to a foreground scope:

```bash
systemd-run --user --scope \
  -p MemoryHigh=22G -p MemoryMax=24G -p MemorySwapMax=2G -p CPUQuota=200% \
  env PHYEVO_PYTHON="$PHYEVO_PYTHON" PHYEVO_DDLP_PYTHON="$PHYEVO_DDLP_PYTHON" \
  bash reproduce.sh --mode full --wait-for-resources
```

Reproduction retains the eight observed frames, six evaluated future frames,
original weights, per-model neural forward counts, and recorded HSR parameters.
No parameter search or training is performed by these commands. For the CPU-only
table checks, install `requirements-metrics.txt` and run
`bash reproduce.sh --mode verify`; no GPU or checkpoints are needed.
