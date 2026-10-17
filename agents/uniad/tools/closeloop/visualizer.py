import cv2 
import numpy as np
import math
from tools.analysis_tools.visualize.mini_run import Visualizer
import glob
import os
import mmcv

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def get_bev_img(outputs, save_fn):
    render_cfg = dict(
        with_occ_map=False,
        with_map=False,
        with_planning=False,
        with_pred_box=True,
        with_pred_traj=True,
        show_command=False,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )
    viser = Visualizer(**render_cfg)
    prediction_dict = viser._parse_predictions(outputs)
    viser.bev_render.reset_canvas(dx=1, dy=1)
    viser.bev_render.set_plot_cfg()

    viser.bev_render.render_pred_box_data(
            prediction_dict['predicted_agent_list'])
    viser.bev_render.render_pred_traj(
            prediction_dict['predicted_agent_list'])
    # viser.bev_render.render_pred_map_data(
    #         prediction_dict['predicted_map_seg'])
    # viser.bev_render.render_pred_box_data(
    #     [prediction_dict['predicted_planning']])
    # viser.bev_render.render_planning_data(
    #     prediction_dict['predicted_planning'], show_command=viser.show_command)
    viser.bev_render.render_sdc_car()
    viser.bev_render.render_legend()
    
    # convert img to byteIO
    # img_buffer = io.BytesIO()
    viser.bev_render.save_fig(save_fn)
    # img_buffer.seek(0)
    
    # # convert byteIO to base64
    # img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # return img_base64

def save_visualize_img(outputs, visual_imgs, save_fn):
    #visiualize
    render_cfg = dict(
        with_occ_map=False,
        with_map=True,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_command=True,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )
    CAM_NAMES = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    viser = Visualizer(**render_cfg)
    prediction_dict = viser._parse_predictions(outputs)
    metas = mmcv.load('data/nus_vis_pose_info.pkl')
    # breakpoint()

    image_dict = {} # TODO: surrounding images
    if visual_imgs is not None:
        visual_imgs = np.array(visual_imgs, dtype=np.float32)
        visual_imgs /= 255.
        temp=np.zeros((6,4,1600,3))
        visual_imgs = np.concatenate([visual_imgs,temp],1)
        image_dict['CAM_FRONT_LEFT'] = visual_imgs[2]
        image_dict['CAM_FRONT'] = visual_imgs[0]
        image_dict['CAM_FRONT_RIGHT'] = visual_imgs[1]
        image_dict['CAM_BACK_RIGHT'] = visual_imgs[5]
        image_dict['CAM_BACK'] = visual_imgs[3]
        image_dict['CAM_BACK_LEFT'] = visual_imgs[4]
    else:
        for cam in CAM_NAMES:
            image_dict[cam] = np.zeros((900, 1600, 3)) # TODO: replace the images
    sample_info = {}
    sample_info['images'] = {}
    sample_info['metas'] = metas
    sample_info['images'] = image_dict
    '''
    sample_info:
        - 'images': 
            'CAM_FRONT': np.array
        - 'metas': 
            'lidar_cs_record'
            'CAM_FRONT':
                'cs_record'
                'imsize'
                'cam_intrinsic'
    }
    '''

    viser.visualize_bev(prediction_dict, save_fn)
    viser.visualize_cam(prediction_dict, sample_info, save_fn)
    viser.combine(save_fn) 
    
def to_video(folder_path, fps=5, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, '*_combine.jpg'))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height//downsample))
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
        
    # media.write_video(os.path.join(folder_path, 'video.mp4'), img_array, fps=10)
    mp4_path = os.path.join(folder_path, 'video.mp4')
    if os.path.exists(mp4_path): 
        os.remove(mp4_path)
    out = cv2.VideoWriter(
        mp4_path, 
        cv2.VideoWriter_fourcc(*'DIVX'), fps, size
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()