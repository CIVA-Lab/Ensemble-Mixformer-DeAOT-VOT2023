import gc
import sys
import vot
import cv2
import numpy as np
import torch

# project path change accordingly
prj_path = '../DeAOT/'
sys.path.append(prj_path)


from model_args import aot_args, sam_args, segtracker_args
from SegTracker import SegTracker
from vot_data_preprocessing import get_bbox


def main():

    handle = vot.VOT("mask", multiobject=True)
    imagefile = handle.frame()
    objects = handle.objects()

    first_img = cv2.imread(imagefile)
    initial_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
    tracks = {}
    for i, gt_mask in enumerate(objects, 1):
        x1, y1, w, h = get_bbox(gt_mask)
        initial_mask[y1:y1+h, x1:x1+w] = gt_mask[y1:y1+h, x1:x1+w] * i
        tracks[i] = []

    seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
    seg_tracker.restart_tracker()

    torch.cuda.empty_cache()
    gc.collect()
    frame_idx = 0

    with torch.cuda.amp.autocast():
        
        while True:
            frame = cv2.imread(imagefile)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                seg_tracker.add_reference(frame, initial_mask)
                torch.cuda.empty_cache()
                gc.collect()
                pred_mask = initial_mask
                frame_idx += 1
                continue
           
            pred_mask = seg_tracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            pred_list = []
            for obj_idx in tracks:
                obj_mask = (pred_mask == obj_idx).astype(np.uint8)
                pred_list.append(obj_mask)
            handle.report(pred_list)

            imagefile = handle.frame()

            if not imagefile:
                break

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
