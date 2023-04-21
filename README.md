# SadTalker (CVPR 2023) on Flyte

This is an attempt towards running [SadTalker inference](https://github.com/Winfredy/SadTalker) on [Flyte](https://github.com/flyteorg/flyte) with just CPUs. To achieve this, the inference code has been adapted from the original SadTalker inference code.

## SadTalker
SadTalker generates 3D motion coefficients (head pose, expression) of the 3DMM from audio and implicitly modulates a novel 3D-aware face render for talking head generation.

## Flyte
Flyte is an orchestrator for data and ML workflows. It's a distributed processing platform that facilitates running highly concurrent workflows.

## Overview
SadTalker's currently hosted on [HuggingFace Spaces](https://huggingface.co/spaces/vinthony/SadTalker) with A10G.
The Flyte flavor was run on a deployed Flyte instance on AWS EKS using CPUs.

The SadTalker inference pipeline was executed on Flyte using the following default model parameters:

```python
@dataclass_json
@dataclass
class ModelParams:
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

| Image | Audio | Model Params | Execution time | Estimated cost | AWS Instance | vCPUs | Memory (GiB) | Actual hourly rate + Flyte deployment costs |
| ----- | ----- | ------------ | -------------- | -------------- | ------------ | ----- | ------------ | ------------------------------------------- |
| ![musk](https://user-images.githubusercontent.com/27777173/233367190-ffed7947-06ec-4609-baad-742ede1327b2.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/spaces/vinthony/SadTalker/blob/main/examples/driven_audio/bus_chinese.wav) (3 sec) | Default args | **6m 23s** [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/adm7fzf5tp98846txhlw?duration=all) |  | g4dn.2xlarge | 8 | 32 | $0.752 + ? |
| ![img_192753_actorpriyankachopra](https://user-images.githubusercontent.com/27777173/233068635-afb950e4-1e04-45af-8e7b-5193a164f5ac.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/examples/driven_audio/chinese_news.wav) (8 sec) | Default args | **9m 58s** [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/atrlrbp7wkv5tflfcgl8?duration=all) | | g4dn.2xlarge | 8 | 32 | $0.752 + ? |
| ![obama](https://user-images.githubusercontent.com/27777173/233065578-cd284886-a756-4323-a404-edcdd62b47b6.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/examples/driven_audio/RD_Radio31_000.wav) (8 sec) | Still=False + Preprocess=Full | **9m 40s** [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/ajmmrngqr2tphtf6c74t?duration=all) | | g4dn.2xlarge | 8 | 32 | $0.752 + ? |
| ![img_192753_actorpriyankachopra](https://user-images.githubusercontent.com/27777173/233068635-afb950e4-1e04-45af-8e7b-5193a164f5ac.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/examples/driven_audio/chinese_news.wav) (8 sec) | Still=True + Enhancer + Preprocess=Full | **19m 8s** [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/a6lqgx7lrb8ls8gf9478?duration=all) | | g4dn.2xlarge | 8 | 32 | $0.752 + ? 
| ![natalie_portman](https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/edPU5HxncLWa1YkgRPNkSd68ONG.jpg) | [![silky-radio-wave](https://user-images.githubusercontent.com/27777173/233053068-eebe0578-069e-49b2-8041-5bfe1ab915c4.png)](https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/audio-oprah-winfrey_95QfotBw.mp3) (25 sec) | Still=False + Enhancer + Preprocess=Full | **56m 3s** [Flyte Demo Link](https://development.uniondemo.run/console/projects/flytesnacks/domains/development/executions/a29jz62n4pdgvx4gxd96?duration=all) | | g4dn.2xlarge | 8 | 32 | $0.752 + ? |

https://user-images.githubusercontent.com/27777173/233373191-80f68163-0f03-4e9f-b469-d22909dfff03.mp4

https://user-images.githubusercontent.com/27777173/233592816-c7237cd0-1828-48d5-ac11-ae3daf7be344.mp4

https://user-images.githubusercontent.com/27777173/233609387-bad59b80-4205-4b55-9c01-168d444cd968.mp4

https://user-images.githubusercontent.com/27777173/233373345-7c7bf076-879d-454e-94a1-39022c779422.mp4

https://user-images.githubusercontent.com/27777173/233592722-df8baf5d-09ea-4521-9ffd-515d7bec45d4.mp4

## Flyte Value Proposition

Several Flyte features have been utilized to optimize the SadTalker inference pipeline:

- **Parallelism**: [Map tasks](https://docs.flyte.org/projects/cookbook/en/latest/auto/core/control_flow/map_task.html) have been employed to execute the code in parallel wherever possible. This approach significantly reduced the execution time.
- **Caching**: Cached one task that analyzes the audio input, but caching opportunities are limited since task outputs are bound to change.
- **Load balancing**: Load balancing is automatic with Flyte since it runs on top of Kubernetes.
- **Scalability**: Flyte can easily handle concurrent requests and scale up or down based on available resources, regardless of the number of executions.
- **Efficient resource usage**: Flyte allows for the [allocation of resources based on task requirements](https://docs.flyte.org/projects/cookbook/en/latest/auto/deployment/customizing_resources.html), ensuring that there is no unnecessary allocation of resources.
