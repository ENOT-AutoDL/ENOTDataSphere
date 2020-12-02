import torch

from mmdet_tools.tools.export_searched_model import extract_model_ckpt_from_search_space

if __name__ == '__main__':
    state_dict = extract_model_ckpt_from_search_space(
        search_space_model_cfg='./configs/wider_face/search_space_ssd.py',
        search_phase_ckpt='PATH_TO_EXPERIMENT/experiments/conv_search_latency/checkpoint-4.pth',
        model_cfg='./configs/wider_face/mobilenet_ssd_from_search_space.py',
    )

    torch.save(state_dict, './searched_mobilenet_state_dict.pth')
