import os
from PIL import Image
import numpy as np
import math
from functools import partial
from omegaconf import OmegaConf
from kerasmodel import KerasModel
# from hugmodel import HugModel
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
from ultralytics.models.sam.modules.tiny_encoder import TinyViT
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
        image_path = self.image_path_list[idx]
        feature_save_path = self.image_path_list[idx].replace('images', 'features').replace('jpg', 'pt')
        image = Image.open(image_path).convert('RGB')
        trans_image_numpy = np.uint8(image)
        # image = cv2.cvtColor(cv2.imread(self.image_path_list[idx]), cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(trans_image_numpy)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        target_feature_torch = torch.load(feature_save_path)
        return self.preprocess(input_image_torch).squeeze(0), target_feature_torch
    
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

def build_tinyvit_model():
    model = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )
    
    ## load pretrained TinyViT weights, please download from https://github.com/wkcn/TinyViT?tab=readme-ov-file
    # pretrained_weights = torch.load("path_to_pth")["model"]
    # del pretrained_weights["head.weight"]
    # del pretrained_weights["head.bias"]
    # model.load_state_dict(pretrained_weights, strict=False)
    
    return model

CONFIG_PATH = 'ultralytics/cfg/training/train_mobile_sam.yaml'
def parse_args():
    return OmegaConf.load(CONFIG_PATH)

def main(args):
    # print(args)
    # model_args = args.model
    train_args = args.train
    # student_model_args = model_args.student_model
    student_model = build_tinyvit_model()
    # student_model.train()

    loss_fn = nn.MSELoss() if train_args.loss_fn == 'mse' else nn.L1Loss()
    # OPTIMIZER = torch.optim.Adam if train_args.optimizer == 'adam' else torch.optim.SGD
    train_dataset = SAMImageDataset(train_args.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size // torch.cuda.device_count(), shuffle=True)

    accelerator = Accelerator(mixed_precision=train_args.mixed_precision, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    # accelerator = Accelerator(mixed_precision=train_args.mixed_precision)
    # optimizer= OPTIMIZER(student_model.parameters(),lr=train_args.lr)
    # optimizer= OPTIMIZER(student_model.parameters(), lr=train_args.lr)
    # optimizer = torch.optim.SGD(student_model.parameters(), lr=train_args.lr, momentum=train_args.momentum, weight_decay=train_args.weight_decay)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=train_args.lr)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer = optimizer,
    #     num_warmup_steps = train_args.num_warmup_steps,
    #     num_training_steps = train_args.epochs * len(train_dataloader) // torch.cuda.device_count(),
    #     eta_min=train_args.eta_min
    # )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_args.epochs * len(train_dataloader), eta_min=train_args.eta_min)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    model = KerasModel(
      net=student_model,
    #   net=nn.ModuleList([student_model,teacher_model]),
      loss_fn = loss_fn,
      metrics_dict = {},
      optimizer=optimizer,
      lr_scheduler=lr_scheduler
    )
    
    dfhistory=model.fit(
                    # num_processes=2,
                    train_data=train_dataloader, 
                    epochs=train_args.epochs, 
                    # learning_rate=train_args.lr,
                    # optimizers=[optimizer, lr_scheduler],
                    # logging_steps=1,
                    # patience=5, 
                    ckpt_path=train_args.ckpt_path,
                    # mixed_precision='fp16',
                    monitor="train_loss",
                    mode="min",
                    plot=True,
                    accelerator=accelerator,
                   )

    

if __name__ == '__main__':
    main(parse_args())