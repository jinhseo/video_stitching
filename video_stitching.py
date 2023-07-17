import argparse
import torch
import PIL
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import itertools
import numpy as np
from torchvision.transforms.functional import to_pil_image
from ocr import ocr

def crop_area(img):
    area = img[:]
    return area

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video stitching tool for full sport video")
    parser.add_argument('--video_path', type=str, default='./',
                        help='path to full video')
    parser.add_argument('--save_path', type=str, default='./',
                        help='path to save clips')
    parser.add_argument('--height', type=int, default=0,
                        help='height to define corner')
    parser.add_argument('--weight', type=int, default=0,
                        help='weight to define corner')
    args = parser.parse_args()
    device = torch.device("cuda")

    vid = torchvision.io.VideoReader(args.video_path)
    md = vid.get_metadata()
    vid_length = int(vid.get_metadata()['video']['duration'][0])
    fps = md['video']['fps'][0]

    recent_frame = [False, False, False, False, False, False, False, False, False, False]
    play_status = [False, False]
    start_time, end_time = 0, 0
    n_th = 0

    corner_h = args.height
    corner_w = args.weight
    for frame in vid:
        if round(frame['pts'] / (1/fps*30), 4).is_integer():
            time_board = to_pil_image(frame['data'][:, corner_w:, corner_h:])
            result, _ = ocr(np.array(time_board))
            print(frame['pts'])
            if len(result) >= 4:
                for r in result.values():
                    if (':' in r[-1] or '.' in r[-1]) or (r[-1].isdigit() and len(r[-1]) >=3):
                        is_play = True
                        recent_frame.append(is_play)
                        recent_frame.pop(0)
                    else:
                        is_play = False
                        recent_frame.append(is_play)
                        recent_frame.pop(0)
            else:
                is_play = False
                recent_frame.append(is_play)
                recent_frame.pop(0)
            if any(recent_frame):
                playing = True
            else:
                playing = False

            play_status.append(playing)
            play_status.pop(0)

            if play_status[0] == False and play_status[-1] == True:
                start_time = frame['pts']
            elif play_status[0] == True and play_status[-1] == False:
                end_time = frame['pts'] - 5
                if start_time > end_time:
                    pass
                elif start_time < end_time:
                    short_clip = torchvision.io.read_video(args.video_path, start_time, end_time, 'sec')
                    torchvision.io.write_video(args.save_path + str(n_th) + '.mp4', short_clip[0], fps)
                    n_th += 1
