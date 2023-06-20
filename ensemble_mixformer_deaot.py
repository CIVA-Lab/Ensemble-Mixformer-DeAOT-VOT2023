
import vot
import cv2
import numpy as np
import sys

# project path change accordingly
prj_path = '/DeAOT/'
sys.path.append(prj_path)

# project path change accordingly
mixformer_path = '/MixFormerSAM/'
sys.path.append(mixformer_path)

from lib.test.evaluation import Tracker
from vot_data_preprocessing import get_bbox, _mask_to_bbox

from model_args import aot_args, sam_args, segtracker_args
from SegTracker import SegTracker

from segment_anything import sam_model_registry, SamPredictor


def _build_init_info(box):
    return {'init_bbox': box}

class mixformerTracker(object):

    def __init__(self, tracker_name='mixformer_vit_online', tracker_param='baseline_large'):

        tracker_params = {'model': 'mixformer_vit_large_online.pth.tar'}
        tracker_info = Tracker(tracker_name, tracker_param, "vot20", tracker_params=tracker_params)
        params = tracker_info.params
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)

    def initialize(self, img_rgb, bbox):

        self.H, self.W, _ = img_rgb.shape
        self.tracker.initialize(img_rgb, _build_init_info(bbox))
        

handle = vot.VOT("mask", multiobject=True)
imagefile = handle.frame()
objects = handle.objects()

# run DeAOT if there are more than 4 objects in the video sequences
if len(objects) > 3:
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
# run MixFormerSAM if there are less than 4 objects in the video sequences
else:
    # path to SAM weights
    sam_checkpoint = "/MixFormerSAM/sam_weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
  
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)

    trackers = [mixformerTracker() for object in objects]

    for ind in range(len(trackers)):
        trackers[ind].initialize(image, _mask_to_bbox(objects[ind]))

    while True:
      imagefile = handle.frame()
      if not imagefile:
          break

      image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
      predictor.set_image(image)
    
      pred_list = []

      for tracker in trackers:
          box = tracker.tracker.track(image)['target_bbox']
          input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
          masks, scores, _ = predictor.predict(point_coords=None,point_labels=None,box=input_box[None, :],multimask_output=True,)
          # get mask with maximum score
          temp = masks[np.argmax(scores)]
          mask = temp.astype(np.uint8)
          pred_list.append(mask)

      handle.report(pred_list)
