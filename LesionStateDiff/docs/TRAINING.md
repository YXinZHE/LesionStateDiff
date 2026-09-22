# Training and generation

Run commands from the repository root after `pip install -e .`.

Variables used below:

```bash
MANIFEST=/path/to/train_manifest.json
OUT=/path/to/outputs
```

Build the training manifest from a patient allow-list:

```bash
python scripts/build_manifest.py \
  --data-root /path/to/fold_root \
  --train-patients /path/to/train_patients.json \
  --output-json "$MANIFEST" \
  --output-csv /path/to/train_manifest.csv
```

Every `EXPECTED_SHA` must be calculated from the exact parent checkpoint:

```bash
sha256sum /path/to/checkpoint.pt
```

## 1. LSDM initial training

```bash
python scripts/train_lsdm.py \
  --parent-checkpoint /path/to/baseline_final.pt \
  --expected-parent-sha256 "$EXPECTED_SHA" \
  --train-manifest "$MANIFEST" \
  --output-dir "$OUT/lsdm_initial" \
  --epochs 20 --batch-size 4 --lr 1e-5 --seed 3
```

## 2. LSDM low-rate refinement

The reported experiment continued from LSDM epoch 15.

```bash
python scripts/train_lsdm_refine.py \
  --base-checkpoint "$OUT/lsdm_initial/checkpoints/epoch_15.pt" \
  --expected-sha256 "$EXPECTED_SHA" \
  --train-manifest "$MANIFEST" \
  --output-dir "$OUT/lsdm_refine" \
  --stage2-epochs 10 --base-total-epoch 15 \
  --batch-size 5 --lr 3e-6 --seed 3
```

## 3. LSDM constant-rate refinement

```bash
python scripts/train_lsdm_constant.py \
  --base-checkpoint "$OUT/lsdm_refine/checkpoints/stage2_epoch5.pt" \
  --expected-sha256 "$EXPECTED_SHA" \
  --train-manifest "$MANIFEST" \
  --stage2-summary "$OUT/lsdm_refine/reports/stage2_training_summary.json" \
  --output-dir "$OUT/lsdm_constant" \
  --stage3-epochs 5 --virtual-epoch-samples 8000 \
  --batch-size 5 --unet-lr 1e-6 --seed 3
```

The region-time map has no trainable parameters. A historical CLI option named
`--region-time-lr` is retained for checkpoint compatibility but is not added
to the optimizer.

## 4. PSC training

The reported PSC run starts from LSDM constant-rate epoch 4.

```bash
python scripts/train_psc.py \
  --base-checkpoint "$OUT/lsdm_constant/checkpoints/stage3_epoch4.pt" \
  --expected-sha256 "$EXPECTED_SHA" \
  --train-manifest "$MANIFEST" \
  --stage3-summary "$OUT/lsdm_constant/reports/stage3_training_summary.json" \
  --output-dir "$OUT/psc" \
  --epochs 5 --virtual-epoch-samples 8000 --batch-size 5 \
  --unet-lr 5e-7 --semantic-lr 5e-5 --seed 3
```

## 5. PSC refinement

The final released lineage refines PSC epoch 3.

```bash
python scripts/train_psc_refine.py \
  --base-checkpoint "$OUT/psc/checkpoints/epoch3.pt" \
  --expected-sha256 "$EXPECTED_SHA" \
  --train-manifest "$MANIFEST" \
  --parent-b3-summary "$OUT/psc/reports/b3_training_summary.json" \
  --b1-validation-dir "$OUT/lsdm_constant/validation_stage3/epoch4" \
  --parent-b3-validation-dir "$OUT/psc/validation_b3/epoch3" \
  --backup-path /path/to/code_snapshot \
  --source-code-root . \
  --output-dir "$OUT/psc_refine" \
  --epochs 5 --virtual-epoch-samples 8000 --batch-size 5 \
  --unet-lr 1e-7 --semantic-lr 1e-5 --seed 3
```

## 6. Audit and generation

Generation is split into an audit/smoke phase and a formal phase.

```bash
COMMON_ARGS="\
  --checkpoint $OUT/psc_refine/checkpoints/semantic_refine_epoch5.pt \
  --data-root /path/to/fold_root \
  --train-patients /path/to/train_patients.json \
  --validation-patients /path/to/validation_patients.json \
  --b3-config $OUT/psc/configs/b3_semantic_config.yaml \
  --output-dir $OUT/formal_generation"

python scripts/generate.py --phase audit $COMMON_ARGS
python scripts/generate.py --phase generate $COMMON_ARGS \
  --per-class 1000 --batch-size 2
```

Do not use held-out test images to construct training-generation pools.
