# Method-to-code map

| Paper component | Implementation |
|---|---|
| Spatial time map | `src/lesionstatediff/region_time.py::build_tau_map` |
| Regional forward diffusion | `region_time.py::q_sample_region` |
| Regional DDIM update | `region_time.py::region_time_ddim_step` |
| Hard masked source | `region_time.py::hard_masked_source` |
| PSC class embedding | `class_semantic_encoder.py::SpatialClassSemanticEncoder` |
| Zero-initialized projection | `class_semantic_encoder.py` |
| Mid-block semantic injection | `semantic_conditioning.py::SemanticConditionedUNet` |
| Final sampler | `semantic_conditioning.py::sample_region_time_ddim_b3` |
| Four-channel U-Net | `four_channel_model_loader.py::build_smis4ch_unet` |
| Manifest dataset | `region_time_dataset.py::RegionTimeDataset` |

The final sampler performs regional reverse diffusion directly. It does not
apply source reinjection or final image blending after denoising.

