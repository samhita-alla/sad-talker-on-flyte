from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import urllib3
import yaml
from flytekit import Resources, dynamic, map_task, task
from flytekit.types.file import FlyteFile
from scipy.spatial import ConvexHull

from ...test_audio2coeff import NamedTextIOWrapper
from .generator import OcclusionAwareSPADEGenerator
from .keypoint_detector import HEEstimator, KPDetector
from .mapping import MappingNet


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
    use_relative_jacobian=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(
            kp_driving_initial["value"][0].data.cpu().numpy()
        ).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
        kp_value_diff *= adapt_movement_scale
        kp_new["value"] = kp_value_diff + kp_source["value"]

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"])
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

    return kp_new


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred * idx_tensor, 1) * 3 - 99
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat(
        [
            torch.ones_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.cos(pitch),
            -torch.sin(pitch),
            torch.zeros_like(pitch),
            torch.sin(pitch),
            torch.cos(pitch),
        ],
        dim=1,
    )
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat(
        [
            torch.cos(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.ones_like(yaw),
            torch.zeros_like(yaw),
            -torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.cos(yaw),
        ],
        dim=1,
    )
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat(
        [
            torch.cos(roll),
            -torch.sin(roll),
            torch.zeros_like(roll),
            torch.sin(roll),
            torch.cos(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.ones_like(roll),
        ],
        dim=1,
    )
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)

    return rot_mat


def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical["value"]  # (bs, k, 3)
    yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if "yaw_in" in he:
        yaw = he["yaw_in"]
    if "pitch_in" in he:
        pitch = he["pitch_in"]
    if "roll_in" in he:
        roll = he["roll_in"]

    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    t, exp = he["t"], he["exp"]
    if wo_exp:
        exp = exp * 0

    # keypoint rotation
    kp_rotated = torch.einsum("bmp,bkp->bkm", rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0] * 0
    t[:, 2] = t[:, 2] * 0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {"value": kp_transformed}


def load_cpk_facevid2vid(
    checkpoint_path,
    generator=None,
    discriminator=None,
    kp_detector=None,
    he_estimator=None,
    optimizer_generator=None,
    optimizer_discriminator=None,
    optimizer_kp_detector=None,
    optimizer_he_estimator=None,
    device="cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if generator is not None:
        generator.load_state_dict(checkpoint["generator"])
    if kp_detector is not None:
        kp_detector.load_state_dict(checkpoint["kp_detector"])
    if he_estimator is not None:
        he_estimator.load_state_dict(checkpoint["he_estimator"])
    if discriminator is not None:
        try:
            discriminator.load_state_dict(checkpoint["discriminator"])
        except:
            print(
                "No discriminator in the state-dict. Dicriminator will be randomly initialized"
            )
    if optimizer_generator is not None:
        optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
    if optimizer_discriminator is not None:
        try:
            optimizer_discriminator.load_state_dict(
                checkpoint["optimizer_discriminator"]
            )
        except RuntimeError as e:
            print(
                "No discriminator optimizer in the state-dict. Optimizer will be not initialized"
            )
    if optimizer_kp_detector is not None:
        optimizer_kp_detector.load_state_dict(checkpoint["optimizer_kp_detector"])
    if optimizer_he_estimator is not None:
        optimizer_he_estimator.load_state_dict(checkpoint["optimizer_he_estimator"])

    return checkpoint["epoch"]


def load_cpk_mapping(
    checkpoint_path,
    mapping=None,
    discriminator=None,
    optimizer_mapping=None,
    optimizer_discriminator=None,
    device="cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if mapping is not None:
        mapping.load_state_dict(checkpoint["mapping"])
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint["discriminator"])
    if optimizer_mapping is not None:
        optimizer_mapping.load_state_dict(checkpoint["optimizer_mapping"])
    if optimizer_discriminator is not None:
        optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])

    return checkpoint["epoch"]


@task(requests=Resources(mem="8Gi", cpu="2"))
def loop_make_animation(
    frame_idx: int,
    target_semantics: torch.Tensor,
    source_image: torch.Tensor,
    generator: OcclusionAwareSPADEGenerator,
    mapping: MappingNet,
    yaw_c_seq: torch.Tensor,
    pitch_c_seq: torch.Tensor,
    roll_c_seq: torch.Tensor,
    kp_canonical: Dict[str, torch.Tensor],
    kp_source: Dict[str, torch.Tensor],
) -> torch.Tensor:
    target_semantics_frame = target_semantics[:, frame_idx]
    he_driving = mapping(target_semantics_frame)

    if torch.count_nonzero(yaw_c_seq).item() != 0:
        he_driving["yaw_in"] = yaw_c_seq[:, frame_idx]
    if torch.count_nonzero(pitch_c_seq).item() != 0:
        he_driving["pitch_in"] = pitch_c_seq[:, frame_idx]
    if torch.count_nonzero(roll_c_seq).item() != 0:
        he_driving["roll_in"] = roll_c_seq[:, frame_idx]

    kp_driving = keypoint_transformation(kp_canonical, he_driving)

    # kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
    # kp_driving_initial=kp_driving_initial)
    kp_norm = kp_driving
    out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
    """
    source_image_new = out['prediction'].squeeze(1)
    kp_canonical_new =  kp_detector(source_image_new)
    he_source_new = he_estimator(source_image_new) 
    kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
    kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
    out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
    """
    return out["prediction"]


@dynamic(requests=Resources(mem="10Gi", cpu="4"))
def make_animation(
    source_image: torch.Tensor,
    source_semantics: torch.Tensor,
    target_semantics: torch.Tensor,
    yaw_c_seq: Optional[torch.Tensor],
    pitch_c_seq: Optional[torch.Tensor],
    roll_c_seq: Optional[torch.Tensor],
    config_path: str,
    device: str,
    free_view_checkpoint: FlyteFile,
    mapping_checkpoint: FlyteFile,
) -> List[torch.Tensor]:
    source_image = source_image.type(torch.FloatTensor)
    source_semantics = source_semantics.type(torch.FloatTensor)
    target_semantics = target_semantics.type(torch.FloatTensor)
    source_image = source_image.to(device)
    source_semantics = source_semantics.to(device)
    target_semantics = target_semantics.to(device)
    if yaw_c_seq is not None:
        yaw_c_seq = yaw_c_seq.type(torch.FloatTensor)
        yaw_c_seq = yaw_c_seq.to(device)
    else:
        yaw_c_seq = None
    if pitch_c_seq is not None:
        pitch_c_seq = pitch_c_seq.type(torch.FloatTensor)
        pitch_c_seq = pitch_c_seq.to(device)
    else:
        pitch_c_seq = None
    if roll_c_seq is not None:
        roll_c_seq = roll_c_seq.type(torch.FloatTensor)
        roll_c_seq = roll_c_seq.to(device)
    else:
        roll_c_seq = None

    http = urllib3.PoolManager()
    response = http.request("GET", config_path, preload_content=False)
    response.auto_close = False
    f = NamedTextIOWrapper(response, name="facerender.yaml")
    config = yaml.safe_load(f)

    generator = OcclusionAwareSPADEGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    kp_extractor = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"],
    )
    he_estimator = HEEstimator(
        **config["model_params"]["he_estimator_params"],
        **config["model_params"]["common_params"],
    )
    mapping = MappingNet(**config["model_params"]["mapping_params"])

    generator.to(device)
    kp_extractor.to(device)
    he_estimator.to(device)
    mapping.to(device)

    for param in generator.parameters():
        param.requires_grad = False
    for param in kp_extractor.parameters():
        param.requires_grad = False
    for param in he_estimator.parameters():
        param.requires_grad = False
    for param in mapping.parameters():
        param.requires_grad = False

    load_cpk_facevid2vid(
        free_view_checkpoint.download(),
        kp_detector=kp_extractor,
        generator=generator,
        he_estimator=he_estimator,
        device=device,
    )

    load_cpk_mapping(mapping_checkpoint.download(), mapping=mapping, device=device)

    kp_extractor.eval()
    generator.eval()
    he_estimator.eval()
    mapping.eval()

    with torch.no_grad():
        list_target_semantics = list(range(target_semantics.shape[1]))
        kp_canonical = kp_extractor(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)
        map_task_partial = partial(
            loop_make_animation,
            target_semantics=target_semantics,
            source_image=source_image,
            generator=generator,
            mapping=mapping,
            yaw_c_seq=yaw_c_seq,
            pitch_c_seq=pitch_c_seq,
            roll_c_seq=roll_c_seq,
            kp_canonical=kp_canonical,
            kp_source=kp_source,
        )
        return map_task(map_task_partial)(frame_idx=list_target_semantics)


class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        source_image = x["source_image"]
        source_semantics = x["source_semantics"]
        target_semantics = x["target_semantics"]
        yaw_c_seq = x["yaw_c_seq"]
        pitch_c_seq = x["pitch_c_seq"]
        roll_c_seq = x["roll_c_seq"]

        predictions_video = make_animation(
            source_image,
            source_semantics,
            target_semantics,
            self.generator,
            self.kp_extractor,
            self.mapping,
            use_exp=True,
            yaw_c_seq=yaw_c_seq,
            pitch_c_seq=pitch_c_seq,
            roll_c_seq=roll_c_seq,
        )

        return predictions_video
