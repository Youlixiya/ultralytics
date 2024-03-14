import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from accelerate import Accelerator

from torch.utils.data import Dataset, DataLoader
from ultralytics.models.edge_sam.utils.transforms import ResizeLongestSide
from ultralytics.models.edge_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

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
        # h, w = image.size[1], image.size[0]
        trans_image_numpy = np.uint8(image)
        # image = cv2.cvtColor(cv2.imread(self.image_path_list[idx]), cv2.COLOR_BGR2RGB)
        input_image = self.transform.apply_image(trans_image_numpy)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        return self.preprocess(input_image_torch).squeeze(0), feature_save_path
    
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

def collate_fn(batch):
    datas = []
    save_paths = []
    for data, save_path in batch:
        datas.append(data)
        save_paths.append(save_path)
    return torch.stack(datas), save_paths
        

def parse_option():
    parser = argparse.ArgumentParser('argument for pre-processing')

    parser.add_argument('--dataset_path', type=str, default="data/sa/images", help='root path of dataset')
    # parser.add_argument('--dataset_dir', type=str, required=True, help='dir of dataset')

    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="ckpts/sam/sam_h.pt")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_option()

    device = args.device
    accelerator = Accelerator()
    device = accelerator.device
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device=device)
    image_encoder = sam.image_encoder
    image_encoder.requires_grad_(False)
    image_encoder.eval()
    dataset = SAMImageDataset(args.dataset_path)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    image_encoder, data_loader = accelerator.prepare(image_encoder, data_loader)

    feature_dir = args.dataset_path.replace('images', 'features')
    # test_image_paths = [os.path.join(test_image_dir, img_name) for img_name in os.listdir(test_image_dir)]
    # test_feature_dir = test_image_dir.replace('images', 'features')
    os.makedirs(feature_dir, exist_ok=True)
    for data, save_path in tqdm(data_loader):
        with torch.no_grad():
          data = data.to(device)
          output = image_encoder(data)
          for i in range(len(save_path)):
              torch.save(output[i], save_path[i])
    # n = len(test_image_paths)
    # for i, test_image_path in enumerate(tqdm(test_image_paths)):
    #     print(i, "/", n)
    #     if ".jpg" in test_image_path:
    #         test_image = cv2.imread(test_image_path)
    #         test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    #         predictor.set_image(test_image)
    #         feature = predictor.features
    #         torch.save(feature.cpu(), test_image_path.replace(".jpg", ".pt").replace('images', 'features'))# .astype(np.float16))