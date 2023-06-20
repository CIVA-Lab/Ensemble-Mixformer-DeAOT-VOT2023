""" This module contains functions for visualization in Jupyter notebooks. """

import os
import io
from threading import Thread, Condition

def is_notebook():
    """ Returns True if the current environment is a Jupyter notebook. 
    
    Returns:
        bool: True if the current environment is a Jupyter notebook.    
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            raise ImportError("console")
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
        if 'VSCODE_PID' in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except ImportError:
        return False
    else:
        return True

if is_notebook():
   
    from IPython.display import display
    from ipywidgets import widgets
    from vot.utilities.draw import ImageDrawHandle

    class SequenceView(object):
        """ A widget for visualizing a sequence. """

        def __init__(self):
            """ Initializes a new instance of the SequenceView class. 
            
            Args:
                sequence (Sequence): The sequence to visualize.
            """

            self._handle = ImageDrawHandle(sequence.frame(0).image())

            self._button_restart = widgets.Button(description='Restart')
            self._button_next = widgets.Button(description='Next')
            self._button_play = widgets.Button(description='Run')
            self._frame = widgets.Label(value="")
            self._frame.layout.display = "none"
            self._frame_feedback = widgets.Label(value="")
            self._image = widgets.Image(value="", format="png", width=sequence.size[0] * 2, height=sequence.size[1] * 2)

            state = dict(frame=0, auto=False, alive=True, region=None)
            condition = Condition()

            self._buttons = widgets.HBox(children=(frame, self._button_restart, self._button_next, button_play, frame2))

        def _push_image(handle):
            """ Pushes an image to the widget. 

            Args:
                handle (ImageDrawHandle): The image handle.
            """
            with io.BytesIO() as output:
                handle.snapshot.save(output, format="PNG")
                return output.getvalue()

    def visualize_tracker(tracker: "Tracker", sequence: "Sequence"):
        """ Visualizes a tracker in a Jupyter notebook.

        Args:
            tracker (Tracker): The tracker to visualize.
            sequence (Sequence): The sequence to visualize.
        """
        from IPython.display import display
        from ipywidgets import widgets
        from vot.utilities.draw import ImageDrawHandle

        def encode_image(handle):
            """ Encodes an image so that it can be displayed in a Jupyter notebook.
            
            Args:
                handle (ImageDrawHandle): The image handle.
            
            Returns:
                bytes: The encoded image."""
            with io.BytesIO() as output:
                handle.snapshot.save(output, format="PNG")
                return output.getvalue()

        handle = ImageDrawHandle(sequence.frame(0).image())

        button_restart = widgets.Button(description='Restart')
        button_next = widgets.Button(description='Next')
        button_play = widgets.Button(description='Run')
        frame = widgets.Label(value="")
        frame.layout.display = "none"
        frame2 = widgets.Label(value="")
        image = widgets.Image(value=encode_image(handle), format="png", width=sequence.size[0] * 2, height=sequence.size[1] * 2)

        state = dict(frame=0, auto=False, alive=True, region=None)
        condition = Condition()

        buttons = widgets.HBox(children=(frame, button_restart, button_next, button_play, frame2))

        image.value = encode_image(handle)

        def run():
            """ Runs the tracker. """

            runtime = tracker.runtime()

            while state["alive"]:

                if state["frame"] == 0:
                    state["region"], _, _ = runtime.initialize(sequence.frame(0), sequence.groundtruth(0))
                else:
                    state["region"], _, _ = runtime.update(sequence.frame(state["frame"]))

                update_image()

                with condition:
                    condition.wait()

                    if state["frame"] == len(sequence):
                        state["alive"] = False
                        continue

                    state["frame"] = state["frame"] + 1


        def update_image():
            """ Updates the image. """
            handle.image(sequence.frame(state["frame"]).image())
            handle.style(color="green").region(sequence.frame(state["frame"]).groundtruth())
            if state["region"]:
                handle.style(color="red").region(state["region"])
            image.value = encode_image(handle)
            frame.value = "Frame: " + str(state["frame"] - 1)

        def on_click(button):
            """ Handles a button click. """
            if button == button_next:
                with condition:
                    state["auto"] = False
                    condition.notify()
            if button == button_restart:
                with condition:
                    state["frame"] = 0
                    condition.notify()
            if button == button_play:
                with condition:
                    state["auto"] = not state["auto"]
                    button.description = "Stop" if state["auto"] else "Run"
                    condition.notify()

        button_next.on_click(on_click)
        button_restart.on_click(on_click)
        button_play.on_click(on_click)
        widgets.jslink((frame, "value"), (frame2, "value"))

        def on_update(_):
            """ Handles a widget update."""
            with condition:
                if state["auto"]:
                    condition.notify()

        frame2.observe(on_update, names=("value", ))

        thread = Thread(target=run)
        display(widgets.Box([widgets.VBox(children=(image, buttons))]))
        thread.start()

    def visualize_results(experiment: "Experiment", sequence: "Sequence"):
        """ Visualizes the results of an experiment in a Jupyter notebook.
        
        Args:
            experiment (Experiment): The experiment to visualize.
            sequence (Sequence): The sequence to visualize.
            
        """

        from IPython.display import display
        from ipywidgets import widgets
        from vot.utilities.draw import ImageDrawHandle

        def encode_image(handle):
            """ Encodes an image so that it can be displayed in a Jupyter notebook.
            
            Args:
                handle (ImageDrawHandle): The image handle.
            
            Returns:
                bytes: The encoded image.
            """

            with io.BytesIO() as output:
                handle.snapshot.save(output, format="PNG")
                return output.getvalue()

        handle = ImageDrawHandle(sequence.frame(0).image())

        button_restart = widgets.Button(description='Restart')
        button_next = widgets.Button(description='Next')
        button_play = widgets.Button(description='Run')
        frame = widgets.Label(value="")
        frame.layout.display = "none"
        frame2 = widgets.Label(value="")
        image = widgets.Image(value=encode_image(handle), format="png", width=sequence.size[0] * 2, height=sequence.size[1] * 2)

        state = dict(frame=0, auto=False, alive=True, region=None)
        condition = Condition()

        buttons = widgets.HBox(children=(frame, button_restart, button_next, button_play, frame2))

        image.value = encode_image(handle)

        def run():
            """ Runs the tracker. """

            runtime = tracker.runtime()

            while state["alive"]:

                if state["frame"] == 0:
                    state["region"], _, _ = runtime.initialize(sequence.frame(0), sequence.groundtruth(0))
                else:
                    state["region"], _, _ = runtime.update(sequence.frame(state["frame"]))

                update_image()

                with condition:
                    condition.wait()

                    if state["frame"] == len(sequence):
                        state["alive"] = False
                        continue

                    state["frame"] = state["frame"] + 1


        def update_image():
            """ Updates the image. """
            handle.image(sequence.frame(state["frame"]).image())
            handle.style(color="green").region(sequence.frame(state["frame"]).groundtruth())
            if state["region"]:
                handle.style(color="red").region(state["region"])
            image.value = encode_image(handle)
            frame.value = "Frame: " + str(state["frame"] - 1)

        def on_click(button):
            """ Handles a button click. """
            if button == button_next:
                with condition:
                    state["auto"] = False
                    condition.notify()
            if button == button_restart:
                with condition:
                    state["frame"] = 0
                    condition.notify()
            if button == button_play:
                with condition:
                    state["auto"] = not state["auto"]
                    button.description = "Stop" if state["auto"] else "Run"
                    condition.notify()

        button_next.on_click(on_click)
        button_restart.on_click(on_click)
        button_play.on_click(on_click)
        widgets.jslink((frame, "value"), (frame2, "value"))

        def on_update(_):
            """ Handles a widget update."""
            with condition:
                if state["auto"]:
                    condition.notify()

        frame2.observe(on_update, names=("value", ))

        thread = Thread(target=run)
        display(widgets.Box([widgets.VBox(children=(image, buttons))]))
        thread.start()