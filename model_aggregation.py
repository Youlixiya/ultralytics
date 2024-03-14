import os
import torch
import argparse
# from mobile_sam import sam_model_registry
from ultralytics.models.edge_sam import sam_model_registry
from ultralytics.models.sam.modules.tiny_encoder import TinyViT
from ultralytics.models.yolo.model import YOLOSAWorld
from ultralytics import SAM

def build_model():
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
    return model

def parse_option():
    parser = argparse.ArgumentParser('argument for model aggregation')

    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--mobile_sam_type', type=str, default="vit_h")
    parser.add_argument('--mobile_sam_ckpt', type=str, default="ckpts/sam/mobile_sam.pt")
    
    parser.add_argument('--save_model_path', type=str, default="./ckpts")
    parser.add_argument('--save_model_name', type=str, default="our_retrained_mobilesam.pt")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_option()

    # our retrained mobile sam 
    print("load model ...")
    # mobile_sam = sam_model_registry[args.mobile_sam_type](checkpoint=args.mobile_sam_ckpt)
    mobile_sam = SAM(args.mobile_sam_ckpt)
    # mobile_sam.image_encoder = build_model()
    student_model = YOLOSAWorld('ckpts/yolov8l-world.pt')
    mobile_sam.model.image_encoder = student_model.sam.image_encoder
    mobile_sam.model.image_encoder.load_state_dict(torch.load(args.ckpt))

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    # torch.save(mobile_sam.state_dict(), os.path.join(args.save_model_path, args.save_model_name))
    torch.save(mobile_sam, os.path.join(args.save_model_path, args.save_model_name))
    print("Completed! The aggregated model is saved as {}".format(os.path.join(args.save_model_path, args.save_model_name)))
