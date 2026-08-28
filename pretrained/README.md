# Pretrained encoder weights

All models use a **PVT-v2-b2** encoder initialized from ImageNet-pretrained weights.

Download `pvt_v2_b2.pth` from the official PVT release
(<https://github.com/whai362/PVT>, "Classification" model zoo) and place it here:

```
pretrained/pvt_v2_b2.pth
```

The model wrappers (`models/pvt_gdla.py`, `models/pvt_da.py`, `models/pvt_sa.py`,
`models/pvt_la.py`) load this file at construction time via the hardcoded relative
path `pretrained/pvt_v2_b2.pth`, so run training/testing from the repository root.
