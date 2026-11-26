# Original data and checkpoints

## Benchmark data

The release data assets are `data_sp.part000` (633,845,760 bytes) and
`data_ie.part000` (722,851,840 bytes): the original 300 SP and 300 IE pairs.
They are losslessly packaged tar streams, not newly rendered replacements.
Download both files and `assets_manifest.json` from the repository's Release
into the same directory, then run:

```bash
python scripts/restore_assets.py /path/to/downloads --only data_sp data_ie
```

The full manifest is also stored in `configs/assets_manifest.json`. Each part
and the reconstructed stream are SHA-256 checked before data are restored.
The restore command refuses to overwrite different existing data.

## Third-party checkpoints

The 13 checkpoint binaries are not committed to Git. Obtain them from the
upstream distributions below and place them at the exact repository-relative
paths in the table. File sizes and complete SHA-256 values are recorded in
`configs/weights.json`. A checkpoint with a different hash must not be substituted
for the recorded file in an exact reproduction. Upstream terms continue to apply;
this release does not assign a new license to their weights. In particular,
OCVP redistribution terms were not established, so its checkpoints are obtained
from upstream rather than rehosted here.

| ID | Upstream distribution | Destination relative to repository root |
|---|---|---|
| `weight_00` | [DDLP OBJ3D](https://github.com/taldatech/ddlp#model-zoo---pretrained-models) | `ddlp/checkpoints/diffuse-ddlp-obj3d128/obj3d128_ddlp.pth` |
| `weight_01` | [DINO-Foresight high-resolution](https://huggingface.co/Sta8is/DINO-Foresight/blob/main/dinof_highres.ckpt) | `third_party/DINO-Foresight/pretrained/dinof_highres.ckpt` |
| `weight_02` | [DINOv2 ViT-B/14 with registers](https://github.com/facebookresearch/dinov2#pretrained-models) | `third_party/dinov2/pretrained/dinov2_vitb14_reg4_pretrain.pth` |
| `weight_03` | [DINO-Foresight PCA](https://huggingface.co/Sta8is/DINO-Foresight/blob/main/dinov2_pca_448_l%5B2_5_8_11%5D_1152.pth) | `third_party/DINO-Foresight/pretrained/dinov2_pca_448_l[2_5_8_11]_1152.pth` |
| `weight_04` | [G-SWM OBJ3D](https://github.com/zhixuan-lin/G-SWM#visualization-with-pretrained-models) | `third_party/G-SWM/pretrained/obj3d.pth` |
| `weight_05` | [OCVP-Seq OBJ3D](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction#evaluation-and-figure-generation) | `third_party/OCVP/pretrained/obj3d/Obj3D/Predictor_OCVPSeq/models/OCVPSeq_obj3d.pth` |
| `weight_06` | [OCVP/SAVi OBJ3D encoder](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction#evaluation-and-figure-generation) | `third_party/OCVP/pretrained/obj3d/Obj3D/models/savi_obj3d.pth` |
| `weight_07` | [SAVi Transformer OBJ3D](https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction#evaluation-and-figure-generation) | `third_party/OCVP/pretrained/obj3d/Obj3D/Predictor_Transformer/models/transformer_obj3d.pth` |
| `weight_08` | [SlotFormer SAVi OBJ3D](https://github.com/pairlab/SlotFormer#reproduce-results) | `third_party/SlotFormer/pretrained/savi_obj3d_params/model_40.pth` |
| `weight_09` | [SlotFormer OBJ3D predictor](https://github.com/pairlab/SlotFormer#reproduce-results) | `third_party/SlotFormer/pretrained/slotformer_obj3d_params/model_200.pth` |
| `weight_10` | [VideoMAEv2 ViT-G hybrid pretraining](https://huggingface.co/OpenGVLab/VideoMAE2/blob/main/mae-g/vit_g_hybrid_pt_1200e.pth) | `VideoMAEv2/checkpoints/vit_g_hybrid_pt_1200e.pth` |
| `weight_11` | [V-JEPA 2 ViT-G](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/blob/main/original/model.pth) | `third_party/vjepa2/vjepa2-vitg-fpc64-256/original/model.pth` |
| `weight_12` | [V-JEPA 2 ViT-L](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/original/model.pth) | `predictive_tracks/sp/checkpoints/vjepa2-vitl-fpc64-256/original/model.pth` |

Use the model's corresponding archived variant; a project may offer several
checkpoints with similar names. The table links to upstream distribution pages;
it does not assert that future upstream downloads will remain byte-identical.
The verifier checks the actual downloaded files:

```bash
python scripts/verify_inputs.py --output input_verification.json
```

This checks all 13 checkpoint hashes and the same 600 input-pair pixel hashes
used in validation, without running a neural network. Fetch OCVP's pinned source
with `python scripts/fetch_ocvp.py` before inference. The source files are verified
against their recorded hashes and are not redistributed in the Git repository.

If you already have the complete 40-part local asset bundle (data and all
checkpoints), `python scripts/restore_assets.py /path/to/release_assets` restores
it in one command. The publicly supplied data files alone are not that full bundle.
