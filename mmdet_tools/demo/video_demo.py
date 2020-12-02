import torch
from mmdet.apis import inference_detector, init_detector
import IPython
import cv2
from io import BytesIO
import time
import PIL
import warnings
import numpy as np
warnings.filterwarnings('ignore')


def show_image(image_array, model, fps_s, format_='jpeg'):
    f = BytesIO()
    result = inference_detector(model, image_array)
    result[0][:, -1] = 0
    img_out = model.show_result(
        image_array,
        result,
        score_thr=0.35,
        show=False,
        font_scale=0.0,
        thickness=3,
        bbox_color='green',
    )
    fps = fps_s[-1] if fps_s else 0.0
    cv2.putText(img_out, str(fps)[:5], (1200, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0))
        
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    pil_img = PIL.Image.fromarray(img_out)
    pil_img.save(f, format=format_)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    
    
def eval_video_stream(video_path, model_path, config_path, device):
    model = init_detector(config_path, model_path, device=device)
    video = cv2.VideoCapture(video_path)
    fpss = video.get(cv2.CAP_PROP_FPS)
    current_frame_num = 0
    fps_s = []
    try:
	    while(True):
	        t1 = time.time()
	        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
	        _, frame = video.read()
	        if frame is None:
	                IPython.display.clear_output(wait=False)
	            print('mean fps:', np.mean(fps_s))
	            break
	        show_image(frame, model, fps_s)
	        t2 = time.time()
	        lasts_frames = (t2 - t1)*fpss
	        current_frame_num += lasts_frames
	        fps_s.append(1 / (t2 - t1))
	        IPython.display.clear_output(wait=True)
    except Exception:
        video.release()
        IPython.display.clear_output(wait=False)
        print("Stream stopped")
    
    
