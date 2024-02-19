import os
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torchkeras
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from ultralytics.models.edge_sam.utils.transforms import ResizeLongestSide
from ultralytics.models.edge_sam import sam_model_registry
from ultralytics import YOLO

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
        return self.preprocess(self.img_size, self.pixel_mean, self.pixel_std, input_image_torch).squeeze(0)
    
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
    
CONFIG_PATH = 'ultralytics/cfg/training/train_yoloworld_sam.yaml'
def parse_args():
    return OmegaConf.load(CONFIG_PATH)

def main(args):
    # print(args)
    model_args = args.model
    train_args = args.train
    student_model_args = model_args.student_model
    teacher_model_args = model_args.teacher_model
    student_model = YOLO(**student_model_args)
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
    optimizer= OPTIMIZER(student_model.adapter.parameters(),lr=train_args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = train_args.num_warmup_steps,
        num_training_steps = train_args.epochs * len(train_dataset)
    )
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