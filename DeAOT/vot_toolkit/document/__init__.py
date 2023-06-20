""" This module contains classes for generating reports and visualizations. """

import typing
from abc import ABC, abstractmethod
import json
import inspect
import threading
import datetime
import collections
import collections.abc
import sys
from asyncio import wait
from asyncio.futures import wrap_future

import numpy as np
import yaml

from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.axes import Axes as PlotAxes
import matplotlib.colors as colors

from attributee import Attributee, Object, Nested, String, Callable, Integer, List

from vot import __version__ as version
from vot import get_logger
from vot.dataset import Sequence
from vot.tracker import Tracker
from vot.analysis import Axes
from vot.utilities import class_fullname
from vot.utilities.data import Grid

class Plot(object):
    """ Base class for all plots. """

    def __init__(self, identifier: str, xlabel: str, ylabel: str,
        xlimits: typing.Tuple[float, float], ylimits: typing.Tuple[float, float], trait = None):
        """ Initializes the plot.
        
        Args:
            identifier (str): The identifier of the plot.
            xlabel (str): The label of the x axis.
            ylabel (str): The label of the y axis.
            xlimits (tuple): The limits of the x axis.
            ylimits (tuple): The limits of the y axis.
            trait (str): The trait of the plot.    
        """

        self._identifier = identifier

        self._manager = StyleManager.default()

        self._figure, self._axes = self._manager.make_figure(trait)

        self._axes.xaxis.set_label_text(xlabel)
        self._axes.yaxis.set_label_text(ylabel)

        if not xlimits is None and not any([x is None for x in xlimits]):
            self._axes.set_xlim(xlimits)
            self._axes.autoscale(False, axis="x")
        if not ylimits is None and not any([y is None for y in ylimits]):
            self._axes.set_ylim(ylimits)
            self._axes.autoscale(False, axis="y")

    def __call__(self, key, data):
        """ Draws the data on the plot."""
        self.draw(key, data)

    def draw(self, key, data):
        """ Draws the data on the plot."""
        raise NotImplementedError
    
    @property
    def axes(self) -> Axes:
        """ Returns the axes of the plot."""
        return self._axes

    def save(self, output, fmt):
        """ Saves the plot to a file."""
        self._figure.savefig(output, format=fmt, bbox_inches='tight', transparent=True)

    @property
    def identifier(self):
        """ Returns the identifier of the plot."""
        return self._identifier

class ScatterPlot(Plot):
    """ A scatter plot."""

    def draw(self, key, data):
        """ Draws the data on the plot. """
        if data is None or len(data) != 2:
            return

        style = self._manager.plot_style(key)
        handle = self._axes.scatter(data[0], data[1], **style.point_style())
        #handle.set_gid("report_%s_%d" % (self._identifier, style["number"]))

class LinePlot(Plot):
    """ A line plot."""

    def draw(self, key, data):
        """ Draws the data on the plot."""
        if data is None or len(data) < 1:
            return

        if isinstance(data[0], tuple):
            # Drawing curve
            if len(data[0]) != 2:
                return
            x, y = zip(*data)
        else:
            y = data
            x = range(len(data))

        style = self._manager.plot_style(key)

        handle = self._axes.plot(x, y, **style.line_style())
       # handle[0].set_gid("report_%s_%d" % (self._identifier, style["number"]))

class ResultsJSONEncoder(json.JSONEncoder):
    """ JSON encoder for results. """

    def default(self, o):
        """ Default encoder. """
        if isinstance(o, Grid):
            return list(o)
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)

class ResultsYAMLEncoder(yaml.Dumper):
    """ YAML encoder for results."""

    def represent_tuple(self, data):
        """ Represents a tuple. """
        return self.represent_list(list(data))


    def represent_object(self, o):
        """ Represents an object. """
        if isinstance(o, Grid):
            return self.represent_list(list(o))
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return self.represent_list(o.tolist())
        else:
            return super().represent_object(o)

ResultsYAMLEncoder.add_representer(collections.OrderedDict, ResultsYAMLEncoder.represent_dict)
ResultsYAMLEncoder.add_representer(tuple, ResultsYAMLEncoder.represent_tuple)
ResultsYAMLEncoder.add_representer(Grid, ResultsYAMLEncoder.represent_object)
ResultsYAMLEncoder.add_representer(np.ndarray, ResultsYAMLEncoder.represent_object)
ResultsYAMLEncoder.add_multi_representer(np.integer, ResultsYAMLEncoder.represent_int)
ResultsYAMLEncoder.add_multi_representer(np.inexact, ResultsYAMLEncoder.represent_float)

def generate_serialized(trackers: typing.List[Tracker], sequences: typing.List[Sequence], results, storage: "Storage", serializer: str):
    """ Generates a serialized report of the results.  """

    doc = dict()
    doc["toolkit"] = version
    doc["timestamp"] = datetime.datetime.now().isoformat()
    doc["trackers"] = {t.reference : t.describe() for t in trackers}
    doc["sequences"] = {s.name : s.describe() for s in sequences}

    doc["results"] = dict()

    for experiment, analyses in results.items():
        exp = dict(parameters=experiment.dump(), type=class_fullname(experiment))
        exp["results"] = []
        for _, data in analyses.items():
            exp["results"].append(data)
        doc["results"][experiment.identifier] = exp

    if serializer == "json":
        with storage.write("results.json") as handle:
            json.dump(doc, handle, indent=2, cls=ResultsJSONEncoder)
    elif serializer == "yaml":
        with storage.write("results.yaml") as handle:
            yaml.dump(doc, handle, Dumper=ResultsYAMLEncoder)
    else:
        raise RuntimeError("Unknown serializer")

def configure_axes(figure, rect=None, _=None):
    """ Configures the axes of the plot. """

    axes = PlotAxes(figure, rect or [0, 0, 1, 1])

    figure.add_axes(axes)

    return axes

def configure_figure(traits=None):
    """ Configures the figure of the plot. """

    args = {}
    if traits == "ar":
        args["figsize"] = (5, 5)
    elif traits == "eao":
        args["figsize"] = (7, 5)
    elif traits == "attributes":
        args["figsize"] = (10, 5)

    return Figure(**args)

class PlotStyle(object):
    """ A style for a plot."""

    def line_style(self, opacity=1):
        """ Returns the style for a line."""
        raise NotImplementedError

    def point_style(self):
        """ Returns the style for a point."""
        raise NotImplementedError

class DefaultStyle(PlotStyle):
    """ The default style for a plot."""

    colormap = get_cmap("tab20b")
    colorcount = 20
    markers = ["o", "v", "<", ">", "^", "8", "*"]

    def __init__(self, number):
        """ Initializes the style. 
        
        Args:
            number (int): The number of the style.
        """
        super().__init__()
        self._number = number

    def line_style(self, opacity=1):
        """ Returns the style for a line.
        
        Args:
            opacity (float): The opacity of the line.
        """
        color = DefaultStyle.colormap((self._number % DefaultStyle.colorcount + 1) / DefaultStyle.colorcount)
        if opacity < 1:
            color = colors.to_rgba(color, opacity)
        return dict(linewidth=1, c=color)

    def point_style(self):
        """ Returns the style for a point.
        
        Args:
            color (str): The color of the point.
            opacity (float): The opacity of the line.
        """
        color = DefaultStyle.colormap((self._number % DefaultStyle.colorcount + 1) / DefaultStyle.colorcount)
        marker = DefaultStyle.markers[self._number % len(DefaultStyle.markers)]
        return dict(marker=marker, c=[color])

class Legend(object):
    """ A legend for a plot."""

    def __init__(self, style_factory=DefaultStyle):
        """ Initializes the legend.
        
        Args:
            style_factory (PlotStyleFactory): The style factory.
        """
        self._mapping = collections.OrderedDict()
        self._counter = 0
        self._style_factory = style_factory

    def _number(self, key):
        """ Returns the number for a key."""
        if not key in self._mapping:
            self._mapping[key] = self._counter
            self._counter += 1
        return self._mapping[key]

    def __getitem__(self, key) -> PlotStyle:
        """ Returns the style for a key."""
        number = self._number(key)
        return self._style_factory(number)

    def _style(self, number):
        """ Returns the style for a number."""
        raise NotImplementedError

    def keys(self):
        """ Returns the keys of the legend."""
        return self._mapping.keys()

    def figure(self, key):
        """ Returns a figure for a key."""
        style = self[key]
        figure = Figure(figsize=(0.1, 0.1))  # TODO: hardcoded
        axes = PlotAxes(figure, [0, 0, 1, 1], yticks=[], xticks=[], frame_on=False)
        figure.add_axes(axes)
        axes.patch.set_visible(False)
        marker_style = style.point_style()
        marker_style["s"] = 40 # Reset size
        axes.scatter(0, 0, **marker_style)
        return figure

class StyleManager(Attributee):
    """ A manager for styles. """

    plots = Callable(default=DefaultStyle)
    axes = Callable(default=configure_axes)
    figure = Callable(default=configure_figure)

    _context = threading.local()

    def __init__(self, **kwargs):
        """ Initializes a new instance of the StyleManager class."""
        super().__init__(**kwargs)
        self._legends = dict()

    def __getitem__(self, key) -> PlotStyle:
        """ Gets the style for the given key."""
        return self.plot_style(key)

    def legend(self, key) -> Legend:
        """ Gets the legend for the given key."""
        if inspect.isclass(key):
            klass = key
        else:
            klass = type(key)

        if not klass in self._legends:
            self._legends[klass] = Legend(self.plots)

        return self._legends[klass]

    def plot_style(self, key) -> PlotStyle:
        """ Gets the plot style for the given key."""
        return self.legend(key)[key]

    def make_axes(self, figure, rect=None, trait=None) -> Axes:
        """ Makes the axes for the given figure."""
        return self.axes(figure, rect, trait)

    def make_figure(self, trait=None) -> typing.Tuple[Figure, Axes]:
        """ Makes the figure for the given trait.
        
        Args:
            trait: The trait for which to make the figure.

        Returns:
            A tuple containing the figure and the axes.
        """
        figure = self.figure(trait)
        axes = self.make_axes(figure, trait=trait)

        return figure, axes

    def __enter__(self):
        """Enters the context of the style manager."""

        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            return self

        StyleManager._context.style_manager = self

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context of the style manager."""
        manager = getattr(StyleManager._context, 'style_manager', None)

        if manager == self:
            StyleManager._context.style_manager = None

    @staticmethod
    def default() -> "StyleManager":
        """ Gets the default style manager."""

        manager = getattr(StyleManager._context, 'style_manager', None)
        if manager is None:
            get_logger().info("Creating new style manager")
            manager = StyleManager()
            StyleManager._context.style_manager = manager

        return manager

class TrackerSorter(Attributee):
    """ A sorter for trackers. """

    experiment = String(default=None)
    analysis = String(default=None)
    result = Integer(val_min=0, default=0)

    def __call__(self, experiments, trackers, sequences):
        """ Sorts the trackers. 
        
        Arguments:
            experiments (list of Experiment): The experiments.
            trackers (list of Tracker): The trackers.
            sequences (list of Sequence): The sequences.
            
        Returns:
            A list of indices of the trackers in the sorted order.
        """
        from vot.analysis import AnalysisError

        if self.experiment is None or self.analysis is None:
            return range(len(trackers))

        experiment = next(filter(lambda x: x.identifier == self.experiment, experiments), None)

        if experiment is None:
            raise RuntimeError("Experiment not found")

        analysis = next(filter(lambda x: x.name == self.analysis, experiment.analyses), None)

        if analysis is None:
            raise RuntimeError("Analysis not found")

        try:

            future = analysis.commit(experiment, trackers, sequences)
            result = future.result()
        except AnalysisError as e:
            raise RuntimeError("Unable to sort trackers", e)

        scores = [x[self.result] for x in result]
        indices = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])]

        return indices

class Generator(Attributee):
    """ A generator for reports."""

    async def generate(self, experiments, trackers, sequences):
        raise NotImplementedError

    async def process(self, analyses, experiment, trackers, sequences):
        if sys.version_info >= (3, 3):
            _Iterable = collections.abc.Iterable
        else:
            _Iterable = collections.Iterable
        if not isinstance(analyses, _Iterable):
            analyses = [analyses]

        futures = []

        for analysis in analyses:
            futures.append(wrap_future(analysis.commit(experiment, trackers, sequences)))

        await wait(futures)

        if len(futures) == 1:
            return futures[0].result()
        else:
            return (future.result() for future in futures)

class ReportConfiguration(Attributee):
    """ A configuration for reports."""

    style = Nested(StyleManager)
    sort = Nested(TrackerSorter)
    generators = List(Object(subclass=Generator), default=[])

# TODO: replace this with report generator and separate json/yaml dump
def generate_document(format: str, config: ReportConfiguration, trackers: typing.List[Tracker], sequences: typing.List[Sequence], results, storage: "Storage"):
    """ Generates a report document.
    
    Args:
        format: The format of the report.
        config: The configuration of the report.
        trackers: The trackers to include in the report.
        sequences: The sequences to include in the report.
        results: The results to include in the report.
        storage: The storage to use for the report.
        
    """

    from .html import generate_html_document
    from .latex import generate_latex_document

    if format == "json":
        generate_serialized(trackers, sequences, results, storage, "json")
    elif format == "yaml":
        generate_serialized(trackers, sequences, results, storage, "yaml")
    else:
        order = config.sort(results.keys(), trackers, sequences)

        with config.style:
            if format == "html":
                generate_html_document(trackers, sequences, results, storage)
            elif format == "latex":
                generate_latex_document(trackers, sequences, results, storage, False, order=order)
            elif format == "pdf":
                generate_latex_document(trackers, sequences, results, storage, True, order=order)

