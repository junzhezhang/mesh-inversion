import os

import torch
import torch.optim

from lib.arguments import Arguments

from lib.utils.common_utils import *
from lib.utils.inversion_dist import *
from lib.data import cub as cub_data

from lib.mesh_inversion import MeshInversion



class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data_module = cub_data
        self.model = MeshInversion(self.args)
        self.dataloader = self.data_module.data_loader(self.args, shuffle=self.args.shuffle)

        # dir of saved results, in .pth file for each instance
        os.makedirs("./outputs", exist_ok=True)
        os.makedirs("./outputs/inversion_results", exist_ok=True)
        os.makedirs(f"./outputs/inversion_results/{args.name}", exist_ok=True)


    def run(self):

        if self.args.use_pred_pose:
            cmr_dict_path = os.path.join(self.args.data_dir, 'cache','cmr_pred_cam.pth')
            cmr_dict = torch.load(cmr_dict_path)

        for i, data in enumerate(self.dataloader):
            
            idx = data['idx'][0].item()

            if self.args.use_pred_pose:
                # replace sfm pose with cmr predicted ones
                img_key, ext = os.path.splitext(os.path.basename(data['img_path'][0]))
                cmr_item = cmr_dict[img_key]
                # to avoid the outliers
                if cmr_item['pred_pose_overlay_iou'] > self.args.filter_noisy_pred_pose:
                    cmr_pred_cam = cmr_item['pred_cam'].unsqueeze(0).type(torch.float32)
                    data['sfm_pose'] = cmr_pred_cam

            self.model.set_target(idx, data, seq=i)
            self.model.init_z()
            self.model.run()
            
            print(f"{idx} completed.")


if __name__ == "__main__":
    args = Arguments(stage='inversion').parser().parse_args()

    trainer = Trainer(args)
    trainer.run()