# Ultralytics YOLO 🚀, AGPL-3.0 license
import numpy as np
import PIL
from PIL.Image import Image
from pathlib import Path
import supervision as sv
import torch
from torch import nn
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils import yaml_load, ROOT
from ultralytics.models.edge_sam.utils.transforms import ResizeLongestSide
from ultralytics.models.edge_sam.build_sam import build_custom_sam
from ultralytics.models.edge_sam.predictor import SamPredictor

class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False, *args, **kwargs):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        stem = Path(model).stem  # filename stem without suffix, i.e. "yolov8n"
        if "-world" in stem:
            # if '-sam' in stem:
            #     new_instance = YOLOWorldSAM(model, *args, **kwargs)
            #     self.__class__ = type(new_instance)
            #     self.__dict__ = new_instance.__dict__
            # else:
            new_instance = YOLOWorld(model)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt") -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model=model, task="detect")

        # Assign default COCO class names
        self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = LayerNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class ConvLoRAMOE(nn.Module):
    def __init__(self, c1, c2, r=64, alpha=128) -> None:
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.encoder = nn.Conv2d(c1, r, 1, bias=False)
        self.gating = nn.Linear(r, 3, bias=False)
        self.experts = nn.ModuleList([
            Conv(r, r, 3),
            Conv(r, r, 5),
            Conv(r, r, 7)
            ]
        )
        self.decoder = nn.Conv2d(r, c2, 1, bias=False)
        self.pooling = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        b = x.shape[0]
        low_rank_x = self.encoder(x)
        gating_score = self.pooling(low_rank_x).softmax(1)
        experts_output = []
        for i, expert in enumerate(self.experts):
            experts_output.append(expert(low_rank_x)*gating_score[:, [i], :, :])
        return self.decoder(sum(experts_output)) * self.alpha

class LoRA(nn.Module):
    def __init__(self, c1, c2, r=64, alpha=128) -> None:
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.encoder = nn.Conv2d(c1, r, 1, bias=False)
        self.decoder = nn.Conv2d(r, c2, 1, bias=False)
    
    def forward(self, x):    
        return self.decoder(self.encoder(x)) * self.alpha
        

class YOLOSAWorldSAImageEncoder(nn.Module):
    # pixel_mean = [123.675, 116.28, 103.53]
    # pixel_std = [58.395, 57.12, 57.375]
    img_size = 1024
    
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone
        # self.backbone.requires_grad_(False)
        self.adapter = adapter
        # self.transform = ResizeLongestSide(self.img_size)
    
    def forward(self, x):
        return self.adapter(self.backbone(x))

class YOLOSAWorldSALoRAImageEncoder(nn.Module):
    # pixel_mean = [123.675, 116.28, 103.53]
    # pixel_std = [58.395, 57.12, 57.375]
    img_size = 1024
    
    def __init__(self, backbone, adapter, r=64, lora_layer_index=[2, 4]):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.lora_modules = nn.ModuleList()
        self.lora_layer_index = lora_layer_index
        for i in range(len(backbone)):
            if i in lora_layer_index:
                c1 = backbone[i].cv1.conv.weight.data.shape[1]
                c2 = backbone[i].cv2.conv.weight.data.shape[0]
                self.lora_modules.append(LoRA(c1, c2, r))
            else:
                self.lora_modules.append(nn.Identity())
        # self.backbone.requires_grad_(False)
        self.adapter = adapter
        # self.transform = ResizeLongestSide(self.img_size)
    
    def forward(self, x):
        for i, module in enumerate(self.backbone):
            if i not in self.lora_layer_index:
                x = module(x)
            else:
                x = module(x) + self.lora_modules[i](x)
        return self.adapter(x)

class YOLOSAWorldSAConvLoRAMOEImageEncoder(nn.Module):
    # pixel_mean = [123.675, 116.28, 103.53]
    # pixel_std = [58.395, 57.12, 57.375]
    img_size = 1024
    
    def __init__(self, backbone, adapter, r=64, lora_layer_index=[2, 4]):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.conv_lora_moe_modules = nn.ModuleList()
        self.lora_layer_index = lora_layer_index
        for i in range(len(backbone)):
            if i in lora_layer_index:
                c1 = backbone[i].cv1.conv.weight.data.shape[1]
                c2 = backbone[i].cv2.conv.weight.data.shape[0]
                self.conv_lora_moe_modules.append(ConvLoRAMOE(c1, c2, r))
            else:
                self.conv_lora_moe_modules.append(nn.Identity())
        # self.backbone.requires_grad_(False)
        self.adapter = adapter
        # self.transform = ResizeLongestSide(self.img_size)
    
    def forward(self, x):
        for i, module in enumerate(self.backbone):
            if i not in self.lora_layer_index:
                x = module(x)
            else:
                x = module(x) + self.conv_lora_moe_modules[i](x)
        return self.adapter(x)
        

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class YOLOSAWorld(YOLOWorld):
    """YOLO-World object detection model."""
    boundingbox_annotator = sv.BoundingBoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
    downscale = 16
    sam_backnone_out_dim = 256
    def __init__(self, model="yolov8s-world.pt", use_lora=False, *args, **kwargs) -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model=model)

        # Assign default COCO class names
        self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")
        if 'sam_ckpt' in kwargs.keys():
            self.sam = torch.load(kwargs['sam_ckpt'])
            # sam_ckpt = torch.load(kwargs['sam_ckpt'])
            # sam_adapter.load_state_dict(sam_ckpt['adapter'])
            # self.sam.load_state_dict(sam_ckpt['prompt_encoder_sam_decoder'])
        else:
            self.set_sam()
            sam_backbone = self.set_sam_backbone()
            sam_adapter = self.set_sam_adapter(sam_backbone)
            if use_lora:
                self.sam.image_encoder = YOLOSAWorldSALoRAImageEncoder(sam_backbone, sam_adapter)
            else:
                self.sam.image_encoder = YOLOSAWorldSAImageEncoder(sam_backbone, sam_adapter)
        # if 'decoder_ckpt' in kwargs.keys():
            # self.set_sam(kwargs['decoder_ckpt'])
            # decoder_ckpt = torch.load(kwargs['decoder_ckpt'])
            # self.sam.load_state_dict(decoder_ckpt)
        
        # self.sam_predictor = SamPredictor(self.sam)
        # else:
        #     sam_ckpt = None 
        # if 'adapter_ckpt' in kwargs.keys():
        #     self.sam_adapter.load_state_dict(torch.load(kwargs['adapter_ckpt']))
    
    def annotate_image(
        self,
        input_image,
        detections,
        categories,
        with_confidence=True,
        ):
        labels = [
            (
                f"{categories[class_id]}: {confidence:.3f}"
                if with_confidence
                else f"{categories[class_id]}"
            )
            for class_id, confidence in
            zip(detections.class_id, detections.confidence)
        ]
        # output_image = MASK_ANNOTATOR.annotate(input_image, detections)
        output_image = self.boundingbox_annotator.annotate(input_image, detections)
        output_image = self.label_annotator.annotate(output_image, detections, labels=labels)
        return output_image
        
    def predict_yoloworld(self, image, text_prompts, pil=True, *args, **kwargs):
        if type(image) != np.ndarray:
            image = np.array(image)
        if type(text_prompts) != list:
            text_prompts = text_prompts.split(',')
        self.set_classes(text_prompts)
        result = self(image, *args, **kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)
        output_image = self.annotate_image(image,
                                           detections,
                                           text_prompts)
        if pil:
            output_image = PIL.Image.fromarray(output_image)
        return result, detections, output_image
    
        

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
    
    def set_sam_backbone(self):
        # sam_backbone = nn.ModuleList()
        # sam_backbone = nn.Sequential()
        sam_backbone = self.model.model[:6]
        # sam_backbone.append(backbone[:2])
        # sam_backbone.append(backbone[2:4])
        # sam_backbone.append(backbone[4:6])
        return sam_backbone
    
    def set_sam_adapter(self, sam_backbone):
        backbone_out_dim = sam_backbone[-1].conv.weight.data.shape[0]
        sam_adapter = nn.Sequential(
            nn.Conv2d(
                backbone_out_dim,
                self.sam_backnone_out_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(self.sam_backnone_out_dim),
            nn.Conv2d(
                self.sam_backnone_out_dim,
                self.sam_backnone_out_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.sam_backnone_out_dim),
            )

        return sam_adapter
    
    # def set_sam_adapter(self, sam_backbone):
    #     sam_adapter = nn.ModuleList()
    #     cur_downscale = 4
    #     for stage in sam_backbone:
    #         stage_out_dim = stage[-1].conv.weight.data.shape[0]
    #         k = self.downscale // cur_downscale
    #         cur_downscale *= 2
    #         sam_adapter.append(nn.Sequential(nn.Conv2d(stage_out_dim, self.sam_backnone_out_dim, kernel_size = k, stride=k),
    #                                               C2f(self.sam_backnone_out_dim, self.sam_backnone_out_dim)))
    #     return sam_adapter
    
    # def sam_backbone_forward(self, x):
    #     output = []
    #     if not isinstance(x, torch.Tensor):
    #         x = self.preprocess(x)
    #     for i in range(len(self.sam_backbone)):
    #         # x = self.sam_adapter[i](self.sam_backnone[i](x))
    #         x = self.sam_backbone[i](x)
    #         # print()
    #         output.append(self.sam_adapter[i](x))
    #     return sum(output)
        # return output
    
    def set_sam(self, sam_ckpt=None):
        self.sam = build_custom_sam(sam_ckpt)
    
    # def preprocess(self, x):
    #     """Normalize pixel values and pad to a square input."""
    #     if isinstance(x, Image):
    #         x = np.array(Image)
    #     x = self.transform.apply_image(x)
    #     x = torch.as_tensor(x, device=self.device)
    #     x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
    #     # Normalize colors
    #     x = (x - self.pixel_mean) / self.pixel_std

    #     # Pad
    #     h, w = x.shape[-2:]
    #     padh = self.img_size - h
    #     padw = self.img_size - w
    #     x = torch.nn.functional.pad(x, (0, padw, 0, padh))
    #     return x

