# Data format

## Directory layout

The generation utility expects a fold root with patient-separated train and
test directories:

```text
fold_root/
  train/
    img/
    mask/
  test/
    img/
    mask/
```

Training itself consumes a JSON manifest rather than scanning the directory.
See `configs/manifest.example.json` for the complete schema.

## Images

- Grayscale, one image per file.
- Original size: `750 x 750`.
- Converted to float in `[-1, 1]`.
- Reflect-padded by nine pixels per side to `768 x 768`.

## Masks

Accepted formats are class-index images, NumPy arrays, or four-channel masks.
Class indices are `0=background, 1=LM, 2=FC, 3=LC, 4=VV`.

For four-channel masks, the required order is:

```text
[LM, FC, LC, VV]
R=LM, G=FC, B=LC, A=VV
```

Masks are zero-padded to `768 x 768`. Overlapping RGBA channels are resolved
in channel order; datasets should therefore be audited for overlaps before
training.

## Split policy

Patient IDs must not overlap across train, validation, and test sets. The
training manifest must contain only training records and must set
`fold1_test_used` to `false` for every row.

Clinical data are not redistributed with this repository. Users are
responsible for data-use approval, de-identification, and a documented split.

