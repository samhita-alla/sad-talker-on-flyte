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
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![musk](https://user-images.githubusercontent.com/27777173/233367190-ffed7947-06ec-4609-baad-742ede1327b2.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/spaces/vinthony/SadTalker/blob/main/examples/driven_audio/bus_chinese.wav) (3 sec) | Default args | 6m 43s [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/ax2lhdnvh4kxrczrnshq?duration=all) |  |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![img_192753_actorpriyankachopra](https://user-images.githubusercontent.com/27777173/233068635-afb950e4-1e04-45af-8e7b-5193a164f5ac.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/spaces/vinthony/SadTalker/blob/main/examples/driven_audio/chinese_news.wav) (8 sec) | Default args | 10m 39s [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/ajbrc7npzgjhmm6wczwj?duration=all) | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![musk](https://user-images.githubusercontent.com/27777173/233367190-ffed7947-06ec-4609-baad-742ede1327b2.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/ariana-grande-7-rings-official-video_23QmiYfu.wav) (1 min) | Still + Preprocess=Full | | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![maxresdefault](https://user-images.githubusercontent.com/27777173/233065578-cd284886-a756-4323-a404-edcdd62b47b6.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/ariana-grande-7-rings-official-video_23QmiYfu.wav) (1 min) | Still + Enhancer | | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![img_192753_actorpriyankachopra](https://user-images.githubusercontent.com/27777173/233068635-afb950e4-1e04-45af-8e7b-5193a164f5ac.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/spaces/vinthony/SadTalker/blob/main/examples/driven_audio/chinese_news.wav) (8 sec) | Still + Enhancer + Preprocess=Full | 20m 41s [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/apcp6chj45sj7ph9jtz4?duration=all) | |
| g4dn.2xlarge | 8 | 32 | $0.752 + ? | ![bb6f6065a9676bda462b93f24fd790368e-17-gal-gadot rsquare w700](https://user-images.githubusercontent.com/27777173/233412184-a9061e02-9ebe-45bf-803f-5a35ca99902d.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/g-eazy-halsey-him-i-mposacoza_PkdvghuB.mp3) (16 sec) | Enhancer + Preprocess=Full | | |



https://user-images.githubusercontent.com/27777173/233373191-80f68163-0f03-4e9f-b469-d22909dfff03.mp4



https://user-images.githubusercontent.com/27777173/233373306-d669b75b-ec6c-4f24-8adf-e78dcc9e8edd.mp4



https://user-images.githubusercontent.com/27777173/233373345-7c7bf076-879d-454e-94a1-39022c779422.mp4




Several Flyte features have been utilized to optimize the SadTalker inference pipeline:

- **Parallelism**: Map tasks have been employed to execute the code in parallel wherever possible. This approach significantly reduced the execution time.
- **Caching**: We cached one task that analyzes the audio input, but caching opportunities are limited since task outputs are bound to change.
- **Load balancing**: Load balancing is automatic with Flyte since it runs on top of Kubernetes, which provides native support for load balancing.
- **Scalability**: Flyte can easily handle concurrent requests and scale up or down based on available resources, regardless of the number of executions.
- **Efficient resource usage**: Flyte enables allocating resources per the task needs, meaning no over allocation of resources.
