import os
from dataclasses import dataclass, field
from typing import List, Optional

import flytekit
from dataclasses_json import dataclass_json
from flytekit import Resources, dynamic, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from .src.facerender.animate import animate_from_coeff_generate_wf
from .src.generate_batch import generate_batch_wf
from .src.generate_facerender_batch import get_facerender_data
from .src.test_audio2coeff import generate_audio_to_coeff
from .src.utils.preprocess import crop_and_extract_wf

import flytekit


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


@task
def init_result_dir(result_dir: str) -> FlyteDirectory:
    working_dir = flytekit.current_context().working_directory
    result_dir_path = os.path.join(working_dir, result_dir)
    os.makedirs(result_dir_path, exist_ok=True)
    with open(os.path.join(result_dir_path, "dummy"), "w") as f:
        f.write("blah blah")
    return FlyteDirectory(result_dir_path, result_dir)


@dynamic(requests=Resources(cpu="3", mem="10Gi", storage="20Gi"))
def sad_talker_dynamic_wf(model_params: ModelParams) -> FlyteFile:
    save_dir = init_result_dir(result_dir=model_params.result_dir)
    checkpoint_dir = model_params.checkpoint_dir
    path_of_lm_croper = os.path.join(
        checkpoint_dir, "shape_predictor_68_face_landmarks.dat"
    )
    path_of_net_recon_model = os.path.join(checkpoint_dir, "epoch_20.pth")
    dir_of_BFM_fitting = os.path.join(checkpoint_dir, "BFM_Fitting.zip")
    wav2lip_checkpoint = os.path.join(checkpoint_dir, "wav2lip.pth")

    audio2pose_checkpoint = os.path.join(checkpoint_dir, "auido2pose_00140-model.pth")

    audio2pose_yaml_path = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/config/auido2pose.yaml"

    audio2exp_checkpoint = os.path.join(checkpoint_dir, "auido2exp_00300-model.pth")
    audio2exp_yaml_path = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/config/auido2exp.yaml"

    free_view_checkpoint = os.path.join(
        checkpoint_dir, "facevid2vid_00189-model.pth.tar"
    )
    if model_params.preprocess == "full":
        mapping_checkpoint = os.path.join(checkpoint_dir, "mapping_00109-model.pth.tar")
        facerender_yaml_path = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/src/config/facerender_still.yaml"
    else:
        mapping_checkpoint = os.path.join(checkpoint_dir, "mapping_00229-model.pth.tar")
        facerender_yaml_path = "https://huggingface.co/spaces/vinthony/SadTalker/raw/main/src/config/facerender.yaml"

    print("3DMM Extraction for source image")
    # flyte wf
    crop_and_extract_wf_output = crop_and_extract_wf(
        path_of_lm_croper=path_of_lm_croper,
        path_of_net_recon_model=path_of_net_recon_model,
        dir_of_BFM_fitting=dir_of_BFM_fitting,
        device=model_params.device,
        input_path=model_params.source_image,
        save_dir=save_dir,
        save_dir_name="first_frame_dir",
        crop_or_resize=model_params.preprocess,
        source_image_flag=True,
    )
    first_coeff_path = crop_and_extract_wf_output.coeff_path
    crop_pic_path = crop_and_extract_wf_output.png_path
    crop_info = crop_and_extract_wf_output.crop_info

    if model_params.ref_eyeblink is not None:
        # flyte wf
        ref_eyeblink_videoname = os.path.splitext(
            os.path.split(model_params.ref_eyeblink)[-1]
        )[0]
        print("3DMM Extraction for the reference video providing eye blinking")
        crop_and_extract_wf_output = crop_and_extract_wf(
            path_of_lm_croper=path_of_lm_croper,
            path_of_net_recon_model=path_of_net_recon_model,
            dir_of_BFM_fitting=dir_of_BFM_fitting,
            device=model_params.device,
            input_path=model_params.ref_eyeblink,
            save_dir=save_dir,
            save_dir_name=ref_eyeblink_videoname,
            crop_or_resize=model_params.preprocess,
        )
        ref_eyeblink_coeff_path = crop_and_extract_wf_output.coeff_path
    else:
        ref_eyeblink_coeff_path = None

    if model_params.ref_pose is not None:
        if model_params.ref_pose == model_params.ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(
                os.path.split(model_params.ref_pose)[-1]
            )[0]
            print("3DMM Extraction for the reference video providing pose")
            crop_and_extract_wf_output = crop_and_extract_wf(
                path_of_lm_croper=path_of_lm_croper,
                path_of_net_recon_model=path_of_net_recon_model,
                dir_of_BFM_fitting=dir_of_BFM_fitting,
                device=model_params.device,
                input_path=model_params.ref_pose,
                save_dir=save_dir,
                save_dir_name=ref_pose_videoname,
                crop_or_resize=model_params.preprocess,
            )
            ref_pose_coeff_path = crop_and_extract_wf_output.coeff_path
    else:
        ref_pose_coeff_path = None

    # audio2ceoff
    # flyte wf
    batch = generate_batch_wf(
        save_dir=save_dir,
        first_coeff_path=first_coeff_path,
        audio_path=model_params.driven_audio,
        device=model_params.device,
        ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
        still=model_params.still,
    )
    # flyte wf
    generate_audio_to_coeff_output = generate_audio_to_coeff(
        audio2pose_checkpoint=audio2pose_checkpoint,
        audio2pose_yaml_path=audio2pose_yaml_path,
        audio2exp_checkpoint=audio2exp_checkpoint,
        audio2exp_yaml_path=audio2exp_yaml_path,
        wav2lip_checkpoint=wav2lip_checkpoint,
        device=model_params.device,
        coeff_save_dir=save_dir,
        pose_style=model_params.pose_style,
        ref_pose_coeff_path=ref_pose_coeff_path,
        indiv_mels=batch.indiv_mels,
        ref=batch.ref_coeff,
        num_frames=batch.num_frames,
        ratio_get=batch.ratio_gt,
        audio_name=batch.audio_name,
        pic_name=batch.pic_name,
    )

    # coeff2video
    data = get_facerender_data(
        save_dir=save_dir,
        coeff_path=generate_audio_to_coeff_output.coeff_path,
        pic_path=crop_pic_path,
        first_coeff_path=first_coeff_path,
        batch_size=model_params.batch_size,
        input_yaw_list=model_params.input_yaw,
        input_pitch_list=model_params.input_pitch,
        input_roll_list=model_params.input_roll,
        expression_scale=model_params.expression_scale,
        still_mode=model_params.still,
        preprocess=model_params.preprocess,
    )

    return animate_from_coeff_generate_wf(
        free_view_checkpoint=free_view_checkpoint,
        mapping_checkpoint=mapping_checkpoint,
        config_path=facerender_yaml_path,
        device=model_params.device,
        video_save_dir=save_dir,
        pic_path=model_params.source_image,
        crop_info=crop_info,
        enhancer=model_params.enhancer,
        source_image=data.source_image,
        source_semantics=data.source_semantics,
        frame_num=data.frame_num,
        target_semantics_list=data.target_semantics_list,
        video_name=data.video_name,
        yaw_c_seq=data.yaw_c_seq,
        pitch_c_seq=data.pitch_c_seq,
        roll_c_seq=data.roll_c_seq,
        audio_path=model_params.driven_audio,
        background_enhancer=model_params.background_enhancer,
        preprocess=model_params.preprocess,
    )


@workflow
def sad_talker_wf(
    model_params: ModelParams = ModelParams(
        # driven_audio="https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/ariana-grande-7-rings-official-videomov_hc2Nmxal.wav",
        # still=False,
        # enhancer="gfpgan",
        preprocess="full",
        driven_audio="https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/examples/driven_audio/RD_Radio31_000.wav",
        source_image="https://user-images.githubusercontent.com/27777173/233065578-cd284886-a756-4323-a404-edcdd62b47b6.jpg",
        # driven_audio="https://huggingface.co/spaces/vinthony/SadTalker/resolve/main/examples/driven_audio/chinese_news.wav",
        # source_image="https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/bb6f6065a9676bda462b93f24fd790368e-17-gal-gadot.rsquare.w700.webp",
    ),
) -> FlyteFile:
    return sad_talker_dynamic_wf(model_params=model_params)


if __name__ == "main":
    print(sad_talker_wf())
