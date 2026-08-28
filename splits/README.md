# Dataset splits

The five benchmarks are redistributed by their original providers, so this repository ships
only the **case-ID lists** needed to rebuild the exact splits used for the reported results.
Each file is one case ID per line, matching the per-case folder names the loaders expect
(see the header comment of the corresponding `cfgs/datasets/*.yaml`).

| Dataset | Files here | How the split was obtained |
|---|---|---|
| PH² | `ph2/train.txt` (80), `ph2/val.txt` (20), `ph2/test.txt` (100) | The full split, exactly as used in the paper. |
| HAM10000 | `ham10000/test.txt` (1015) | The held-out test set the reported HAM10000 numbers are measured on. The remaining images form train/val. |
| BUSI | none | Random 80/10/10 split over the full dataset; no fixed ID list was kept. |
| Synapse | none | [TransUNet](https://github.com/Beckschen/TransUNet) protocol: 18 training volumes, 12 evaluation volumes. Use the lists shipped with TransUNet. |
| ACDC | none | Standard 2D-slice preprocessing; evaluation is on the 40 held-out ED/ES volumes. |

On Synapse the 12 evaluation volumes double as the validation set during training, with
`VAL_DIR` and `TEST_DIR` pointing at the same directory in `cfgs/datasets/synapse.yaml`,
following the TransUNet protocol this benchmark is conventionally reported under.
