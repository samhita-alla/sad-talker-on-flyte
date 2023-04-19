# SadTalker (CVPR 2023) on Flyte

This is an attempt towards running [SadTalker inference](https://github.com/Winfredy/SadTalker) on [Flyte](https://github.com/flyteorg/flyte) with just CPUs.

## SadTalker
SadTalker generates 3D motion coefficients (head pose, expression) of the 3DMM from audio and implicitly modulates a novel 3D-aware face render for talking head generation.

## Flyte
Flyte is an orchestrator for data and ML workflows. It's a distributed processing platform for highly concurrent workflows.

## Overview
SadTalker's currently hosted on [HuggingFace Spaces](https://huggingface.co/spaces/vinthony/SadTalker) with A10G.
The Flyte flavor was run on a deployed Flyte instance on AWS EKS using CPUs.

The SadTalker inference pipeline was executed on Flyte using the following default model parameters:

```python
@dataclass_json
@dataclass
class ModelParams:
    driven_audio: FlyteFile = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/examples/driven_audio/bus_chinese.wav"
    source_image: FlyteFile = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/examples/source_image/full_body_2.png"
    ref_pose: Optional[str] = None
    ref_eyeblink: Optional[str] = None
    result_dir: str = "results"
    pose_style: int = 0
    batch_size: int = 2
    expression_scale: float = 1.0
    input_yaw: List[int] = field(default_factory=lambda: [0])
    input_pitch: List[int] = field(default_factory=lambda: [0])
    input_roll: List[int] = field(default_factory=lambda: [0])
    enhancer: str = ""
    background_enhancer: str = "realesrgan"
    device: str = "cpu"
    still: bool = True
    preprocess: str = "crop"
    net_recon: str = "resnet50"
    use_last_fc: bool = False
    focal: float = 1015.0
    center: float = 112.0
    camera_d: float = 10.0
    z_near: float = 5.0
    z_far: float = 15.0
    bfm_folder: str = "./checkpoints/BFM_Fitting/"
    bfm_model: str = "BFM_model_front.mat"
    checkpoint_dir: str = (
        "https://github.com/Winfredy/SadTalker/releases/download/v0.0.2"
    )
```

The table below shows the estimated cost, execution time, and resources used for running SadTalker on Flyte:

| AWS Instance | vCPUs | Memory (GiB) | Actual hourly rate + Flyte deployment costs | Image | Audio | Model Params | Execution time | Estimated cost |
| ------------ | ----- | ------------ | ------------------------------------------- | ----- | ----- | ------------ | -------------- | -------------- |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![screen-shot-2022-10-28-at-9-51-32-am-1666965104](https://user-images.githubusercontent.com/27777173/233045031-a3ce76e2-4898-45d9-b5cf-660d56c61ca9.png) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://user-images.githubusercontent.com/27777173/233049773-18a4f6a4-30bd-4c88-b2f5-5eeb80c452ae.mov) (8 sec) | Still + Preprocess=Crop | | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![screen-shot-2022-10-28-at-9-51-32-am-1666965104](https://user-images.githubusercontent.com/27777173/233045031-a3ce76e2-4898-45d9-b5cf-660d56c61ca9.png) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/datasets/Samhita/SadTalkerData/raw/main/ariana-grande-7-rings-official-video_23QmiYfu.wav) (1 min) | Still + Preprocess=Full | | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | | | Still + Enhancer | | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | | | Still + Enhancer + Preprocess=Full | | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | | | Enhancer + Preprocess=Full | | |

Several Flyte features have been utilized to optimize the SadTalker inference pipeline:

- **Parallelism**: Map tasks have been employed to execute the code in parallel wherever possible. This approach significantly reduced the execution time.
- **Caching**: We cached one task that analyzes the audio input, but caching opportunities are limited since task outputs are bound to change.
- **Load balancing**: Load balancing is automatic with Flyte since it runs on top of Kubernetes, which provides native support for load balancing.
- **Scalability**: Flyte can easily handle concurrent requests and scale up or down based on available resources, regardless of the number of executions.
- **Efficient resource usage**: Flyte enables allocating resources per the task needs, meaning no over allocation of resources.
