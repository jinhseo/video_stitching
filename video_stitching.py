import argparse
import torch
import PIL
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import itertools
import json
import numpy as np
from torchvision.transforms.functional import to_pil_image
from ocr import ocr
from EasyOCR.easyocr import easyocr
from collections import Counter
#import preprocessing_json

symbols = [':', '.', '-', ',']

def preprocessing_json(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    quaters = ['Q1', 'Q2', 'Q3', 'Q4']

    empty_list = [[],[],[],[]]
    record_data = [[], [], [], []]
    for i, q in enumerate(quaters):
        for actions in json_data[q]:
            act_code = int(actions['action_code'])
            play_min = str(actions['play_min'])
            play_sec = str(actions['play_sec'])
            #if act_code == 201 or act_code == 202:
            if act_code > 200:
                #if act_code == 201:
                #    save_act = 0
                #elif act_code == 202:
                #    save_act = 1

                if len(play_sec) == 1:
                    play_sec = play_sec.zfill(2)
                play_time = int(play_min + play_sec)
                #empty_list[i].append([play_time, act_code])
                record_data[i].append([str(play_time), str(act_code)])
    return record_data

def run_ocr(last_frames, corner_w, corner_h):
    last_results = []
    #import IPython; IPython.embed()
    #ocr(torch.stack(last_frames))
    #import IPython; IPython.embed()
    for last_frame in last_frames:
        #import IPython; IPython.embed()
        #time_board = to_pil_image(last_frame)
        #result, _ = ocr(np.array(time_board)) ###np.array(time_board).shape == (180, 320, 3)
        result, _ = ocr(last_frame)
        update_result(last_results, result)
    return last_results

def update_result(result):
    #for r in result.values():
    #    if (any(item in r[-1] for item in symbols) and (len(r[-1]) < 5)) or (r[-1].isdigit() and len(r[-1]) >= 3):
    #    #if ((':' or '.' or '-') in r[-1] and (len(r[-1]) < 5)) or (r[-1].isdigit() and len(r[-1]) >= 3):
    #        last_results.append(r[-1])
    last_results = []
    for r in result:
        if len(r) >= 3 and ' ' in r[-1]:
            last_results.append(r[-1].split(' ')[0])
        elif (len(r) >= 5 and any(item in r[-2] for item in symbols)) or (len(r) >=5 and r[-2].isdigit() and len(r[-2]) >= 3):
            last_results.append(r[-2])
        elif (len(r) >= 5 and any(item in r[-1] for item in symbols)) or (len(r) >=5 and r[-1].isdigit() and len(r[-1]) >= 3):
            last_results.append(r[-1])

    return last_results

def voting(last_results, final_result, play_status):
    vote_box = []
    if len(last_results) >= 3:
        vote_box = []
        elements = list(set(last_results))
        for last_result in last_results:
            vote_box.append(elements.index(last_result))
        final_result = elements[max(set(vote_box), key=vote_box.count)]
        play_status.append(True)
        play_status.pop(0)
    else:
        final_result = 0
        play_status.append(False)
        play_status.pop(0)
    return final_result, play_status

def to_digit(game_clock):
    for item in symbols:
        no_time = "".join(game_clock.split(item))
        if no_time.isdigit():
            return no_time
    return None

def refine_order(revise):
    revised = []
    for i, (k, v) in enumerate(outlier_time.items()):
        if i == 0:
            s = int(revise[0])
            e = int(k)
        else:
            s = before_s
            e = int(k)
        descending = list(range(e, s+1))
        descending.reverse()
        revised.append(descending)
        revised.append([e] * (v[-1] - v[0] - 1))
        before_s = int(k)
    revised = list(itertools.chain.from_iterable(revised))
    revised = [r for r in revised if int(str(r)[1]) <= 5]
    return revised

def write_anno():
    anno = 0
    return anno

def playing():
    return
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
    parser.add_argument('--width', type=int, default=0,
                        help='width to define corner')
    parser.add_argument('--json_path', type=str, default='./211219.json',
                        help='path to json file')

    args = parser.parse_args()
    device = torch.device("cuda")

    vid = torchvision.io.VideoReader(args.video_path)
    md = vid.get_metadata()
    vid_length = int(vid.get_metadata()['video']['duration'][0])
    fps = md['video']['fps'][0]
    torchvision.set_video_backend('video_reader')
    play_status = [False, False]
    start_time, end_time, current_time = 0, 0, 0
    n_th = 0
    save_fps = 15

    symbols = [':', '.', '-', ',']
    corner_h = args.height
    corner_w = args.width
    last_frames = []
    final_result = 0
    play_status = [False, False]
    quarter = 0
    final_results = []
    recording = False
    clock_memory = ['', '']
    quarter_info = []
    #reader = easyocr.Reader(['ko','en'])
    reader = easyocr.Reader(['ko'])
    record_data = preprocessing_json(args.json_path)

    for i, frame in enumerate(vid.seek(200)):
        if (i+5)%6 == 0:
            last_frames = []
            quarter_info = []
        else:
            #last_frames.append(frame['data'][:,corner_w:,corner_h:])
            last_frames.append(frame['data'][:, 553:656, 1023:1207])
            quarter_info.append(frame['data'][:, 627:655, 1023:1077]) ## quarter
        if frame['pts'] > 200 and round(frame['pts'] / (1/fps*30), 4).is_integer():
            print(frame['pts'])
            result = reader.readtext_batched(np.array(torch.stack(last_frames).permute(0,2,3,1)), detail=0)
            if result[0]:
                #quarter = reader.readtext_batched(np.array(frame['data'][:, 627:655, 1023:1077].unsqueeze(0).permute(0,2,3,1)), detail=0)
                quarter = reader.readtext_batched(np.array(torch.stack(quarter_info).permute(0,2,3,1)), detail=0)
                quarter = set(list(itertools.chain.from_iterable(quarter)))
                #quarter = update_result(quarter)

            #quarter = update_result(quarter)
            last_results = update_result(result)
            final_result, play_status = voting(last_results, final_result, play_status)

            if play_status[0] == False and play_status[-1] == True:
                start_time = frame['pts']
                recording = True
            elif play_status[0] == True and play_status[-1] == False:
                end_time = frame['pts']
                recording = False
                if start_time <= end_time:# and current_time <= start_time:
                    current_time = end_time
                    revise = []
                    for f in final_results:
                        for item in symbols:
                            no_time = "".join(f.split(item))
                            if no_time.isdigit():
                                revise.append(no_time)
                                break
                            #else:
                            #    revise.append(no_time)
                    play_time = str(revise[0])
                    check = Counter(revise)
                    outlier = dict(Counter({k: c for k, c in check.items() if c >= 5}))
                    outlier_time = dict()
                    for o_v in outlier.keys():
                        arg_min = np.where(np.array(revise) == o_v)[0][0]
                        arg_max = np.where(np.array(revise) == o_v)[0][-1]
                        outlier_time[o_v] = [arg_min, arg_max]

                    print('start from: ' + str(start_time) + 'to' + str(end_time))
                    quarter_info = int(quarter[0][0][0])
                    sub_folder = './clips_img/' + str(quarter_info) + '/' + str(n_th) + '_' + revise[0] + '_' + revise[-1]
                    os.makedirs(sub_folder, exist_ok=True)
                    short_clip = torchvision.io.read_video(args.video_path, start_time, end_time, 'sec', output_format='TCHW')
                    revised = refine_order(revise)
                    '''revised = []
                    for i, (k, v) in enumerate(outlier_time.items()):
                        if i == 0:
                            s = int(revise[0])
                            e = int(k)
                        else:
                            s = before_s
                            e = int(k)
                        descending = list(range(e, s+1))
                        descending.reverse()
                        revised.append(descending)
                        revised.append([e] * (v[-1] - v[0] - 1))
                        before_s = int(k)
                    revised = list(itertools.chain.from_iterable(revised))
                    revised = [r for r in revised if int(str(r)[1]) <= 5]
                    '''
                    revised = list(itertools.chain.from_iterable([[item] * round(save_fps) for item in revised]))
                    skip_point = [item[0] for item in record_data[quarter_info-1] if item[1] == '216' or item[1] == '225']
                    for s_c in range(0, len(short_clip[0]), round(fps/save_fps)):
                        n_index = str(int((s_c + 2)/2))
                        torchvision.io.write_png(short_clip[0][s_c], sub_folder + '/' + n_index + '.png')
                    file_name = sub_folder + '/annotations.txt'
                    f = open(file_name, 'w')
                    anno_file = sub_folder.split('/')[-1]
                    for r_d in record_data[quarter_info - 1]:
                        offset_time = int(r_d[0]) + 2
                        if int(revise[-1]) <= offset_time <= int(revise[0]):
                               index_in_folder = revised.index(offset_time)
                               label = int(r_d[1][1:])
                               f.write(anno_file + ' ' + str(index_in_folder - 7) + ' ' + str(index_in_folder + 8) + ' ' + str(label) + '\n')
                    f.close()
                    #import IPython; IPython.embed()
                    '''for s_c in range(0, len(short_clip[0]), round(fps/save_fps)):
                        for r_d in record_data[quarter_info-1]:
                            offset_time = int(r_d[0]) + 2
                            if (int(revise[-1]) <= offset_time <= int(revise[0])) and r_d[1] != 216:
                                n_index = str(int((s_c + 2)/2))
                                torchvision.io.write_png(short_clip[0][s_c], sub_folder + '/' + n_index + '.png')
                    '''
                    '''outlier_duration = []
                    for i, (k, v) in enumerate(outlier_time.items()):
                        outlier_duration.append(v[1] - v[0])
                        if i == 0:
                            end_time = start_time + v[0]
                        else:
                            end_time = start_time + v[0] - outlier_duration[i-1]
                        import IPython; IPython.embed()
                        short_clip = torchvision.io.read_video(args.video_path, start_time, end_time, 'sec', output_format='TCHW')
                        sub_folder = './clips_img/' +str(quarter_info) + '/' + str(n_th) + '_' + play_time + '_' + k
                        os.makedirs(sub_folder, exist_ok=True)
                        for s_c in range(0, len(short_clip[0]), round(fps/save_fps)):
                            torchvision.io.write_png(short_clip[0][s_c], sub_folder + '/' + str(s_c) + '.png')
                        file_name = sub_folder + '/annotations.txt'
                        f = open(file_name, 'w')
                        anno_file = sub_folder.split('/')[-1]
                        for r_d in record_data[quarter_info-1]:
                            offset_time = int(r_d[0]) + 2
                            if int(k) <= offset_time <= int(play_time):
                                index_in_folder = (int(play_time) - offset_time)*save_fps
                                label = int(r_d[1][1:])
                                f.write(anno_file + ' ' + str(index_in_folder - 7) + ' ' + str(index_in_folder + 8) + ' ' + str(label) + '\n')
                        f.close()
                        start_time = start_time + v[1]
                        play_time = k
                    #short_clip = torchvision.io.read_video(args.video_path, start_time, end_time, 'sec', output_format='TCHW')
                    #sub_folder = './clips_img/' + str(n_th) + '_' + revise[0] + '_' + revise[-1]
                    #os.makedirs(sub_folder, exist_ok=True)
                    #for s_c in range(0, len(short_clip[0]), 2):
                    #    torchvision.io.write_png(short_clip[0][s_c], sub_folder + '/' + str(s_c) + '.png')
                    #for i, vid_img in enumerate(short_clip[0]):
                    #    os.makedirs(sub_folder, exist_ok=True)
                    #    torchvision.io.write_png(vid_img, sub_folder + '/' + str(i) + '.png', compression_level=3)
                    '''
                    #file_name = sub_folder + '/video_annotation.txt'
                    #with open(file_name, 'w+') as file:
                    #    file.write('\n'.join(revise))
                    #import IPython; IPython.embed()
                    #torchvision.io.write_video(args.save_path + '/kbl_' + str(n_th) + '.mp4', short_clip[0], fps/2)
                    n_th += 1
                    #import IPython; IPython.embed()

            if recording:
                final_results.append(final_result)
            else:
                final_results = []


    import IPython; IPython.embed()
