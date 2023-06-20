import os
import sys
import glob
import cv2
import numpy as np
import vot

from vot_data_preprocessing import get_bbox, get_mask

import DeAOT_tracker
import mixformer_tracker as mixformer
import stark_tracker as stark

DeAot = DeAOT_tracker.DeAOTTracker(DeAOT_tracker.segtracker_args, DeAOT_tracker.sam_args, DeAOT_tracker.aot_args)

handle = vot.VOT("mask", multiobject=True)
imagefile = handle.frame()
objects = handle.objects()
frame = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

DeAot.initialize(frame, objects)

strak_trackers = [stark.starkTracker() for object in objects]
mixformer_trackers = [mixformer.mixformerTracker() for object in objects]


for ind in range(len(objects)):
    strak_trackers[ind].initialize(frame, get_bbox(objects[ind]))
    mixformer_trackers[ind].initialize(frame, get_bbox(objects[ind]))


while True:

    frame = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    pred_list1 = DeAot.track(frame)

    mixformer.predictor.set_image(frame)

    pred_list = []

    for ind in range(len(objects)):

        box1 = strak_trackers[ind].tracker.track(frame)['target_bbox']
        input_box1 = np.array([box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]])
        masks1, _, _ = mixformer.predictor.predict(point_coords=None,point_labels=None,box=input_box1[None, :],multimask_output=False,)
        mask1 = masks1[0].astype(np.uint8)


        box = mixformer_trackers[ind].tracker.track(frame)['target_bbox']
        input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
        masks, _, _ = mixformer.predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=False,)
        mask = masks[0].astype(np.uint8)

        pred_list.append((mask == 1) | (mask1==1) | (pred_list1[ind] == 1))


    pred_mask = []
    for mk in pred_list:
        
        refined_box = get_bbox(mk)
        input_box = np.array([refined_box[0], refined_box[1], refined_box[0]+refined_box[2], refined_box[1]+refined_box[3]])
        masks, _, _ = mixformer.predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=False,)
        mask_vf = masks[0].astype(np.uint8)

        pred_mask.append(mask_vf)


    handle.report(pred_mask)

    imagefile = handle.frame()
    if not imagefile:
        break







