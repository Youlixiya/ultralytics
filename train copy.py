import os
from PIL import Image
import numpy as np
import math
from functools import partial
from omegaconf import OmegaConf
import torchkeras
# from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from ultralytics.models.edge_sam.utils.transforms import ResizeLongestSide
from ultralytics.models.edge_sam import sam_model_registry
from ultralytics.models.yolo.model import YOLOSAWorld

class SAMImageDataset(Dataset):
    img_size = 1024
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __init__(self,
                 image_dir,
                 ):
        self.image_file_list = os.listdir(image_dir)
        self.image_path_list = [os.path.join(image_dir, i) for i in self.image_file_list]
        # self.image_list = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in self.image_path_list]
        self.transform = ResizeLongestSide(self.img_size)
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx]).convert('RGB')
        # h, w = image.size[1], image.size[0]
        trans_image_numpy = np.uint8(image)
        # image = cv2.cvtColor(cv2.imread(self.image_path_list[idx]), cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(trans_image_numpy)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        return self.preprocess(input_image_torch).squeeze(0)
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = torch.nn.functional.pad(x, (0, padw, 0, padh))
        return x

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float,
    eta_min: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, eta_min:int = 0, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        eta_min=eta_min
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# class YOLOSAWorldSAImageEncoder(nn.Module):
#     # pixel_mean = [123.675, 116.28, 103.53]
#     # pixel_std = [58.395, 57.12, 57.375]
#     img_size = 1024
    
#     def __init__(self, backbone, adapter):
#         super().__init__()
#         self.backbone = backbone
#         self.adapter = adapter
#         # self.transform = ResizeLongestSide(self.img_size)
    
#     def forward(self, x):
#         return self.adapter(self.backbone(x))

CONFIG_PATH = 'ultralytics/cfg/training/train_yolosa_world_conv_lora_moe.yaml'
def parse_args():
    return OmegaConf.load(CONFIG_PATH)

def main(args):
    # print(args)
    model_args = args.model
    train_args = args.train
    student_model_args = model_args.student_model
    teacher_model_args = model_args.teacher_model
    student_model = YOLOSAWorld(**student_model_args)
    student_model = student_model.sam.image_encoder
    # backbone = student_model.set_sam_backbone()
    # adapter = student_model.set_sam_adapter(backbone)
    # student_model = YOLOSAWorldSAImageEncoder(backbone, adapter)
    # student_model.load_state_dict(torch.load('ckpts/yolov8l-world-sam.pt'))
    # student_model.__call__ = student_model.sam_backbone_forward
    teacher_model = sam_model_registry[teacher_model_args.sam_model_type](checkpoint=teacher_model_args.sam_checkpoint).image_encoder
    # teacher_model = model_registry[teacher_model_args.tap_model_type](checkpoint=teacher_model_args.tap_checkpoint).image_encoder
    # student_model.tiny_clip_image_encoder.requires_grad_(False)
    teacher_model.requires_grad_(False)
    # student_model = student_model.half()
    # teacher_model = student_model.half()
    loss_fn = nn.MSELoss() if train_args.loss_fn == 'mse' else nn.L1Loss()
    OPTIMIZER = torch.optim.Adam if train_args.optimizer == 'adam' else torch.optim.SGD
    train_dataset = SAMImageDataset(train_args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)

    accelerator = Accelerator(mixed_precision=train_args.mixed_precision, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    # accelerator = Accelerator(mixed_precision=train_args.mixed_precision)
    # optimizer= OPTIMIZER(student_model.parameters(),lr=train_args.lr)
    optimizer= OPTIMIZER(list(student_model.adapter.parameters())+list(student_model.lora_modules.parameters()), lr=train_args.lr)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer = optimizer,
    #     num_warmup_steps = train_args.num_warmup_steps,
    #     num_training_steps = train_args.epochs * len(train_dataloader) // torch.cuda.device_count(),
    #     eta_min=train_args.eta_min
    # )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_args.epochs * len(train_dataloader), eta_min=train_args.eta_min)
    model = torchkeras.KerasModel(
      student_model=student_model,
      teacher_model=teacher_model,
    #   net=nn.ModuleList([student_model,teacher_model]),
      loss_fn = loss_fn,
      optimizer= optimizer,
      lr_scheduler=lr_scheduler,
      metrics_dict = {},
    )
    
    dfhistory=model.fit(
                    # num_processes=2,
                    train_data=train_dataloader, 
                    epochs=train_args.epochs, 
                    # patience=5, 
                    ckpt_path=train_args.save_path,
                    # mixed_precision='fp16',
                    monitor="train_loss",
                    mode="min",
                    plot=True,
                    accelerator=accelerator
                   )

    

if __name__ == '__main__':
    main(parse_args())