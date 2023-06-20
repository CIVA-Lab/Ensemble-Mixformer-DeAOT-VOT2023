""" Dummy sequences for testing purposes."""

import os
import math
import tempfile

from vot.dataset import BasedSequence
from vot.region import Rectangle
from vot.region.io import write_trajectory
from vot.utilities import write_properties

from PIL import Image
import numpy as np

def _generate(base, length, size, objects):
    """Generate a new dummy sequence.
    
    Args:
        base (str): The base directory for the sequence.
        length (int): The length of the sequence.
        size (tuple): The size of the sequence.
        objects (int): The number of objects in the sequence.
    """

    background_color = Image.fromarray(np.random.normal(15, 5, (size[1], size[0], 3)).astype(np.uint8))
    background_depth = Image.fromarray(np.ones((size[1], size[0]), dtype=np.uint8) * 200)
    background_ir = Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8))

    template = Image.open(os.path.join(os.path.dirname(__file__), "cow.png"))

    dir_color = os.path.join(base, "color")
    dir_depth = os.path.join(base, "depth")
    dir_ir = os.path.join(base, "ir")

    os.makedirs(dir_color, exist_ok=True)
    os.makedirs(dir_depth, exist_ok=True)
    os.makedirs(dir_ir, exist_ok=True)

    path_color = os.path.join(dir_color, "%08d.jpg")
    path_depth = os.path.join(dir_depth, "%08d.png")
    path_ir = os.path.join(dir_ir, "%08d.png")

    groundtruth = {i : [] for i in range(objects)}

    center_x = size[0] / 2
    center_y = size[1] / 2

    radius = min(center_x - template.size[0], center_y - template.size[1])

    speed = (math.pi * 2) / length
    offset = (math.pi * 2) / objects

    for i in range(length):
        frame_color = background_color.copy()
        frame_depth = background_depth.copy()
        frame_ir = background_ir.copy()

        for o in range(objects):

            x = int(center_x + math.cos(i * speed + offset * o) * radius - template.size[0] / 2)
            y = int(center_y + math.sin(i * speed + offset * o) * radius - template.size[1] / 2)

            frame_color.paste(template, (x, y), template)
            frame_depth.paste(10, (x, y), template)
            frame_ir.paste(240, (x, y), template)

            groundtruth[o].append(Rectangle(x, y, template.size[0], template.size[1]))

        frame_color.save(path_color % (i + 1))
        frame_depth.save(path_depth % (i + 1))
        frame_ir.save(path_ir % (i + 1))

    if objects == 1:
        write_trajectory(os.path.join(base, "groundtruth.txt"), groundtruth[0])
    else:
        for i, g in groundtruth.items():
            write_trajectory(os.path.join(base, "groundtruth_%03d.txt" % i), g)

    metadata = {"name": "dummy", "fps" : 30, "format" : "dummy",
                        "channel.default": "color"}
    write_properties(os.path.join(base, "sequence"), metadata)

def generate_dummy(length=100, size=(640, 480), objects=1):
        """Create a new dummy sequence.
        
        Args:
            length (int, optional): The length of the sequence. Defaults to 100.
            size (tuple, optional): The size of the sequence. Defaults to (640, 480).
            objects (int, optional): The number of objects in the sequence. Defaults to 1.
        """
        from vot.dataset import load_sequence

        base = os.path.join(tempfile.gettempdir(), "vot_dummy_%d_%d_%d_%d" % (length, size[0], size[1], objects))
        if not os.path.isdir(base) or not os.path.isfile(os.path.join(base, "sequence")):
            _generate(base, length, size, objects)

        return load_sequence(base)
