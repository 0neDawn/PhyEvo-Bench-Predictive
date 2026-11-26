# Third-party materials

The release retains the original notices; a repository-level license must not
be interpreted as replacing third-party terms. No new blanket license has been
assigned to the authors' benchmark/HSR code in this preparation package.

| Component | Source | Source-code license |
|---|---|---|
| DDLP | https://github.com/taldatech/ddlp | MIT |
| VideoMAEv2 | https://github.com/OpenGVLab/VideoMAEv2 | MIT |
| SlotFormer | https://github.com/pairlab/SlotFormer | MIT |
| NERV | https://github.com/pairlab/nerv | MIT |
| G-SWM | https://github.com/zhixuan-lin/G-SWM | GPL-3.0 |
| DINO-Foresight | https://github.com/Sta8is/DINO-Foresight | MIT |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Apache-2.0 |
| JEPA intuitive physics | https://github.com/facebookresearch/jepa-intuitive-physics | CC BY-NC 4.0 |
| V-JEPA 2 | https://github.com/facebookresearch/vjepa2 | MIT |
| OCVP | https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction | No license found in the pinned checkout |

Exact available revisions are in `configs/third_party.json`. Individual files
are hashed in `audit/source_manifest.json`. Most files came from the final
experiment source snapshots. The external V-JEPA runtime and NERV dependency
were captured during packaging; the V-JEPA predictor hash additionally matches
the historical experiment record. The manifest distinguishes these sources.

OCVP's 34 required Python files are not included in the public source archive.
`scripts/fetch_ocvp.py` retrieves those exact files from the original repository
at the recorded commit and checks their archived hashes. Local validation may
restore the same files from the private audit snapshot; those local files remain
excluded from Git and the public source archive.

Checkpoint files are separate assets, identified by byte count and SHA-256 in
`configs/weights.json`. Their use and redistribution remain subject to the
checkpoint authors' terms, which may differ from the source-code license.
Packaging a local checkpoint is not a new permission to redistribute it. In
particular, OCVP checkpoint redistribution terms were not established here;
use the upstream distribution when those terms are unresolved.

Original upstream licenses are retained alongside included code. Local absolute
paths and interpreter locations were adapted in release copies; numeric model
and scoring operations were retained. Original archived source hashes and
release hashes are both listed for inspection. G-SWM's release entry verifies
the recorded upstream revision and packaged source hashes instead of requiring
a nested `.git` directory; this does not change inference or scoring operations.
