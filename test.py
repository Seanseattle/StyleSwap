import argparse
import cv2
import numpy as np
import os
import paddle
from models.styleswap import FullGenerator
from models.arcface import Backbone

from utils.align_face import back_matrix, dealign, align_source, align_target
from utils.util import paddle2cv, cv2paddle
# from utils.prepare_data import LandmarkModel


def image_test(args, generator, id_network):
    target_img = cv2.imread(args.target_img_path.replace('images', 'images512'))
    # target_img = cv2.imread(args.target_img_path)
    source_img = cv2.imread(args.source_img_path)
    if args.align_source or args.align_target:
        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    
    theta_for_id = None
    if args.align_source:
        target_lm = landmarkModel.get(target_img)
        source_lm = landmarkModel.get(source_img)
        M = align_source(source_lm)
        aligned_source = cv2.warpAffine(source_img, M, (224, 224), flags=cv2.INTER_LINEAR)
    else:
        aligned_source = source_img
        if os.path.exists(args.source_img_path.replace('images', 'theta')[:-4] + ".npy"):
            theta_for_id = args.source_img_path.replace('images', 'theta')[:-4] + ".npy"
            theta_for_id = paddle.to_tensor(np.load(theta_for_id), dtype='float32').unsqueeze(axis=0)
    
    if args.align_target:
        aligned_target = align_target(target_lm, size=args.size)
    else:
        aligned_target = target_img

    aligned_source = cv2paddle(aligned_source)
    aligned_target = cv2paddle(aligned_target)

    with paddle.no_grad():
        # id_emb = id_network(aligned_source)
        id_emb = id_network(aligned_source, theta_for_id)
        result = generator(aligned_target, id_emb)[:2]
        result = paddle2cv(result)
        cv2.imwrite(args.output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def create_id_network():
    id_network = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    id_network.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
    id_network.eval()
    return id_network


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleSwap Test")
    parser.add_argument('--source_img_path', type=str, default='./examples/images/2.png')
    parser.add_argument('--target_img_path', type=str, default='./examples/images/12012.png')
    parser.add_argument('--output_path', type=str, default='results/test.png', help='path to the output path')
    parser.add_argument('--size', type=int, default=512, help='size of the test images')
    parser.add_argument('--align_source', type=bool, default=False, help='align the source image')
    parser.add_argument('--align_target', type=bool, default=False, help='align the target image')
    # parser.add_argument("--latent", type=int, default=512)
    # parser.add_argument("--n_mlp", type=int, default=8)
    # parser.add_argument("--channel_multiplier", type=int, default=2)
    # parser.add_argument("--narrow", type=float, default=1)
    args = parser.parse_args()
    print(args)
    if args.size == 256:
        generator = FullGenerator(args.size, 512, 8, channel_multiplier=1, narrow=0.5, outter_mask=True)
        generator.set_dict(paddle.load('./checkpoints/styleswap256.pdparams'))
    else:
        generator = FullGenerator(args.size, 512, 8, channel_multiplier=2, narrow=1, outter_mask=False)
        generator.set_dict(paddle.load('./checkpoints/styleswap512.pdparams'))
    id_network = create_id_network()
    image_test(args, generator, id_network)



  




