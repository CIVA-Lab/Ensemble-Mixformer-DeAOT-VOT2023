import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def driver_img_seq(tracker_name, tracker_param, img_seq, output_dir, optional_box=None, debug=None, 
                   save_results=False, obj_num = "1", sam_model=None, tracker_params=None):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video", tracker_params=tracker_params, sam_model=sam_model)
    tracker.run_img_seq(img_seq_path=img_seq, output_dir=output_dir, tracker_name=tracker_name, 
                        optional_box=optional_box, debug=debug, obj_num=obj_num, save_results=save_results)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('img_seq', type=str, help='path to an image sequence.')
    parser.add_argument('--output_dir', type=str, help='path to an root image sequence.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=True)

    parser.add_argument('--obj_num', type=str, default='1', help='specify object number that should be tracked, if only one object in the scene just give 1.')
    parser.add_argument('--sam_model', type=str, default=None, help='Which SAM (Segment Anything Model) model to use.')

    parser.add_argument('--params__model', type=str, default=None, help="Tracking model path.")
    parser.add_argument('--params__update_interval', type=int, default=None, help="Update interval of online tracking.")
    parser.add_argument('--params__online_sizes', type=int, default=None)
    parser.add_argument('--params__search_area_scale', type=float, default=None)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)
    parser.add_argument('--params__vis_attn', type=int, choices=[0, 1], default=0, help="Whether visualize the attention maps.")

    args = parser.parse_args()

    tracker_params = {}
    for param in list(filter(lambda s: s.split('__')[0] == 'params' and getattr(args, s) != None, args.__dir__())):
        tracker_params[param.split('__')[1]] = getattr(args, param)
    print(tracker_params)

    driver_img_seq(args.tracker_name, args.tracker_param, args.img_seq, args.output_dir, args.optional_box, args.debug, args.save_results, 
                   args.obj_num, args.sam_model, tracker_params=tracker_params)


if __name__ == '__main__':
    main()
