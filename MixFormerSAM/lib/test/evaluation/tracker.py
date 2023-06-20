import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

SAM_MODELS = {
  'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
  'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
  'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
}


def get_model(model_name):
  """Downloads a model to ./sam_weights if it does not exist and return path to it.

  Args:
      model_name (str): SAM model name. One of ['vit_h', 'vit_l', 'vit_b'].
  """
  url = SAM_MODELS[model_name]
  filename = url.split('/')[-1]
  filepath = os.path.join('./sam_weights', filename)
  if not os.path.exists(filepath):
    os.makedirs('./sam_weights', exist_ok=True)
    print(f'Downloading {url} to {filepath}')
    r = requests.get(url, allow_redirects=True)
    open(filepath, 'wb').write(r.content)

  return filepath


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, tracker_params=None, sam_model=None):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        if sam_model is not None:
            sam_checkpoint = get_model(sam_model)
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint).cuda()
            self.sam_predictor = SamPredictor(sam)
        else:
            self.sam_predictor = None

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        self.params = self.get_parameters(tracker_params)

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        # params = self.get_parameters()
        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        # params = self.get_parameters()
        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            raise NotImplementedError("We haven't support cv_show now.")
            # while True:
            #     # cv.waitKey()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
            #                1.5, (0, 0, 0), 1)
            #
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)
            #     break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            #              (0, 255, 0), 5)
            #
            # font_color = (0, 0, 0)
            # cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)

            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)
            #
            #     # cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_img_seq(self, img_seq_path, output_dir, tracker_name, optional_box=None, debug=None, visdom_info=None, obj_num="1", save_results=False):
        """Run the tracker on an image sequence.
        args:
            debug: Debug level.
        """
        # create output dir
        res_dir = os.path.join(output_dir, tracker_name)
        if not os.path.exists(res_dir):
                os.makedirs(res_dir)

        # create output dir
        mask_folder = 'masks-' + obj_num
        mask_dir = os.path.join(res_dir, mask_folder)
        if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)

        params = self.params

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output_boxes = []

        img_list_full = []
        _, _, files = next(os.walk(img_seq_path))
        files = sorted(files) 
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_list_full.append(img_seq_path + file)
        n_frames = len(img_list_full) 
        print('Number of frames in input directory: {}'.format(n_frames))

        display_name = 'Display: ' + tracker.params.tracker_name
        frame = cv.imread(img_list_full[0], cv.IMREAD_COLOR) 

        def _build_init_info(box):
            return {'init_bbox': box}

        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            raise NotImplementedError("We haven't support cv_show now.")

        for fr in range(1, n_frames):  # start from the second frame
            frame = cv.imread(img_list_full[fr], cv.IMREAD_COLOR)
            print(img_list_full[fr])

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            if self.sam_predictor is not None:
              self.sam_predictor.set_image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
              box_xyxy = np.array(state)
              box_xyxy[2:] += box_xyxy[:2]
              masks, _, _ = self.sam_predictor.predict(box=box_xyxy)

              #get max mask
              mask = masks[0].astype(np.uint8)
              mask*=255

              mask_name = os.path.basename(img_list_full[fr]).replace('jpg', 'png')
              out_mask_path = os.path.join(mask_dir, mask_name)
              cv.imwrite(out_mask_path, mask)


        if save_results:
            bboxs_filename = 'bboxs-' + obj_num
            base_results_path = os.path.join(res_dir, bboxs_filename)

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self, tracker_params=None):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        search_area_scale = None
        if tracker_params is not None and 'search_area_scale' in tracker_params:
            search_area_scale = tracker_params['search_area_scale']
        model = ''
        if tracker_params is not None and 'model' in tracker_params:
            model = tracker_params['model']
        params = param_module.parameters(self.parameter_name, model, search_area_scale)
        if tracker_params is not None:
            for param_k, v in tracker_params.items():
                setattr(params, param_k, v)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



