
""" TraX protocol implementation for the toolkit. TraX is a communication protocol for visual object tracking.
 It enables communication between a tracker and a client. The protocol was originally developed for the VOT challenge to address
 the need for a unified communication interface between trackers and benchmarking tools.
"""
import sys
import os
import time
import re
import subprocess
import shutil
import shlex
import socket as socketio
import tempfile
import logging
import unittest
from typing import Tuple
from threading import Thread, Lock

import numpy as np

import colorama

from trax import TraxException
from trax.client import Client
from trax.image import FileImage
from trax.region import Region as TraxRegion
from trax.region import Polygon as TraxPolygon
from trax.region import Mask as TraxMask
from trax.region import Rectangle as TraxRectangle

from vot.dataset import Frame, DatasetException
from vot.region import Region, Polygon, Rectangle, Mask
from vot.tracker import Tracker, TrackerRuntime, TrackerException, Objects, ObjectStatus
from vot.utilities import to_logical, to_number, normalize_path

PORT_POOL_MIN = 9090
PORT_POOL_MAX = 65535

logger = logging.getLogger("vot")

class LogAggregator(object):
    """ Aggregates log messages from the tracker. """

    def __init__(self):
        """ Initializes the aggregator."""
        self._fragments = []

    def __call__(self, fragment):
        """ Appends a new fragment to the log."""
        self._fragments.append(fragment)

    def __str__(self):
        """ Returns the aggregated log."""
        return "".join(self._fragments)

class ColorizedOutput(object):
    """ Colorized output for the tracker."""

    def __init__(self):
        """ Initializes the colorized output."""
        colorama.init()

    def __call__(self, fragment):
        """ Prints a new fragment to the output.
        
        Args:
            fragment: The fragment to be printed.
        """
        print(colorama.Fore.CYAN + fragment + colorama.Fore.RESET, end="")

class PythonCrashHelper(object):
    """ Helper class for detecting Python crashes in the tracker."""

    def __init__(self):
        """ Initializes the crash helper."""
        self._matcher = re.compile(r'''
            ^Traceback
            [\s\S]+?
            (?=^\[|\Z)
            ''', re.M | re.X)

    def __call__(self, log, directory):
        """ Detects Python crashes in the log.
        
        Args:
            log: The log to be checked.
            directory: The directory where the log is stored.
        """
        matches = self._matcher.findall(log)
        if len(matches) > 0:
            return matches[-1].group(0)
        return None

def convert_frame(frame: Frame, channels: list) -> dict:
    """ Converts a frame to a dictionary of Trax images.

    Args:
        frame: The frame to be converted.
        channels: The list of channels to be converted.

    Returns:
        A dictionary of Trax images.
    """
    tlist = dict()

    for channel in channels:
        image = frame.filename(channel)
        if image is None:
            raise DatasetException("Frame does not have information for channel: {}".format(channel))

        tlist[channel] = FileImage.create(image)

    return tlist

def convert_region(region: Region) -> TraxRegion:
    """ Converts a region to a Trax region.
    
    Args:
        region: The region to be converted.
        
    Returns:
        A Trax region.
    """
    if isinstance(region, Rectangle):
        return TraxRectangle.create(region.x, region.y, region.width, region.height)
    elif isinstance(region, Polygon):
        return TraxPolygon.create([region[i] for i in range(region.size)])
    elif isinstance(region, Mask):
        return TraxMask.create(region.mask, x=region.offset[0], y=region.offset[1])
    return None

def convert_traxregion(region: TraxRegion) -> Region:
    """ Converts a Trax region to a region.

    Args:
        region: The Trax region to be converted.

    Returns:
        A region.
    """
    if region.type == TraxRegion.RECTANGLE:
        x, y, width, height = region.bounds()
        return Rectangle(x, y, width, height)
    elif region.type == TraxRegion.POLYGON:
        return Polygon(list(region))
    elif region.type == TraxRegion.MASK:
        return Mask(region.array(), region.offset(), optimize=True)
    return None

def convert_objects(objects: Objects) -> TraxRegion:
    """ Converts a list of objects to a Trax region.

    Args:
        objects: The list of objects to be converted.

    Returns:    
        A Trax region.
    """
    if objects is None: return []
    if isinstance(objects, (list, )):
        return [(convert_region(o.region), dict(o.properties)) for o in objects]
    if isinstance(objects, (ObjectStatus, )):
        return [(convert_region(objects.region), dict(objects.properties))]
    else:
        return [(convert_region(objects), dict())]

def convert_traxobjects(region: TraxRegion) -> Region:
    """ Converts a Trax region to a region.

    Args:
        region: The Trax region to be converted.

    Returns:
        A region.

    """
    if region.type == TraxRegion.RECTANGLE:
        x, y, width, height = region.bounds()
        return Rectangle(x, y, width, height)
    elif region.type == TraxRegion.POLYGON:
        return Polygon(list(region))
    elif region.type == TraxRegion.MASK:
        return Mask(region.array(), region.offset(), optimize=True)
    return None

class TestRasterMethods(unittest.TestCase):
    """ Tests for the raster methods. """

    def test_convert_traxregion(self):
        """ Tests the conversion of Trax regions."""
        convert_traxregion(TraxRectangle.create(0, 0, 10, 10))
        convert_traxregion(TraxPolygon.create([(0, 0), (10, 0), (10, 10), (0, 10)]))
        convert_traxregion(TraxMask.create(np.ones((100, 100), dtype=np.uint8)))

    def test_convert_region(self):
        """ Tests the conversion of regions."""
        convert_region(Rectangle(0, 0, 10, 10))
        convert_region(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
        convert_region(Mask(np.ones((100, 100), dtype=np.uint8)))

def open_local_port(port: int):
    """ Opens a local port for listening."""
    socket = socketio.socket(socketio.AF_INET, socketio.SOCK_STREAM)
    try:
        socket.setsockopt(socketio.SOL_SOCKET, socketio.SO_REUSEADDR, 1)
        socket.bind(('127.0.0.1', port))
        socket.listen(1)
        return socket
    except OSError:
        try:
            socket.close()
        except OSError:
            pass
        return None

def normalize_paths(paths, tracker):
    """ Normalizes a list of paths relative to the tracker source."""
    root = os.path.dirname(tracker.source)
    return [normalize_path(path, root) for path in paths]

class TrackerProcess(object):
    """ A tracker process. This class is used to run trackers in a separate process and handles
     starting, stopping and communication with the process. """
    
    def __init__(self, command: str, envvars=dict(), timeout=30, log=False, socket=False):
        """ Initializes a new tracker process.

        Args:
            command: The command to run the tracker.
            envvars: A dictionary of environment variables to be set for the tracker process.
            timeout: The timeout for the tracker process.
            log: Whether to log the tracker output.
            socket: Whether to use a socket for communication.

        """
        environment = dict(os.environ)
        environment.update(envvars)

        self._workdir = tempfile.mkdtemp()

        self._returncode = None
        self._socket = None

        if socket:
            for port in range(PORT_POOL_MIN, PORT_POOL_MAX):
                socket = open_local_port(port)
                if not socket is None:
                    self._socket = socket
                    break
            environment["TRAX_SOCKET"] = "{}".format(port)

        logger.debug("Running process: %s", command)

        if sys.platform.startswith("win"):
            self._process = subprocess.Popen(
                    command,
                    cwd=self._workdir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment, bufsize=0, close_fds=False)
        else:
            self._process = subprocess.Popen(
                    shlex.split(command),
                    shell=False,
                    cwd=self._workdir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment, bufsize=0, close_fds=False)

        self._timeout = timeout
        self._client = None

        self._watchdog_lock = Lock()
        self._watchdog_counter = 0
        self._watchdog = Thread(target=self._watchdog_loop)
        self._watchdog.start()

        self._watchdog_reset(True)

        try:
            if socket:
                self._client = Client(stream=self._socket.fileno(), timeout=30, log=log)
            else:
                self._client = Client(
                    stream=(self._process.stdin.fileno(), self._process.stdout.fileno()), log=log
                )
                
        except TraxException as e:
            self.terminate()
            self._watchdog_reset(False)
            raise e
        self._watchdog_reset(False)

        self._has_vot_wrapper = not self._client.get("vot") is None
        self._multiobject = self._client.get("multiobject")

    def _watchdog_reset(self, enable=True):
        """ Resets the watchdog.

        Args:
            enable: Whether to enable the watchdog.

        """
        if self._watchdog_counter == 0:
            return

        if enable:
            self._watchdog_counter = self._timeout * 10
        else:
            self._watchdog_counter = -1

    def _watchdog_loop(self):
        """ The watchdog loop. This loop is used to monitor the tracker process and terminate it if it does not respond anymore."""

        while self.alive:
            time.sleep(0.1)
            if self._watchdog_counter < 0:
                continue
            self._watchdog_counter = self._watchdog_counter - 1
            if not self._watchdog_counter:
                logger.warning("Timeout reached, terminating tracker")
                self.terminate()
                break

    @property
    def has_vot_wrapper(self):
        """ Whether the tracker has a VOT wrapper. VOT wrapper limits TraX functionality and injects a property at handshake to let the client know this."""
        return self._has_vot_wrapper

    @property
    def returncode(self):
        """ The return code of the tracker process."""
        return self._returncode

    @property
    def workdir(self):
        """ The working directory of the tracker process."""
        return self._workdir

    @property
    def interrupted(self):
        """ Whether the tracker process was interrupted."""
        return self._watchdog_counter == 0

    @property
    def alive(self):
        """ Whether the tracker process is alive."""
        if self._process is None:
            return False
        self._returncode = self._process.returncode
        return self._returncode is None

    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """ Initializes the tracker. This method is used to initialize the tracker with the first frame. It returns the initial state of the tracker.

        Args:
            frame: The first frame.
            new: The initial state of the tracker.
            properties: The properties to be set for the tracker.

        Returns:
            The initial state of the tracker.

        Raises:
            TraxException: If the tracker is not alive.
        """

        if not self.alive:
            raise TraxException("Tracker not alive")

        if properties is None:
            properties = dict()

        tlist = convert_frame(frame, self._client.channels)
        tobjects = convert_objects(new)

        self._watchdog_reset(True)

        status, elapsed = self._client.initialize(tlist, tobjects, properties)

        self._watchdog_reset(False)

        status = [ObjectStatus(convert_traxregion(region), properties) for region, properties in status]

        return status, elapsed


    def update(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """ Updates the tracker with a new frame. This method is used to update the tracker with a new frame. It returns the new state of the tracker.

        Args:
            frame: The new frame.
            new: The new state of the tracker.
            properties: The properties to be set for the tracker.

        Returns:
            The new state of the tracker.

        Raises:
            TraxException: If the tracker is not alive.

        """

        if not self.alive:
            raise TraxException("Tracker not alive")

        tlist = convert_frame(frame, self._client.channels)

        tobjects = convert_objects(new)

        self._watchdog_reset(True)

        status, elapsed = self._client.frame(tlist, properties, tobjects)

        self._watchdog_reset(False)

        status = [ObjectStatus(convert_traxregion(region), properties) for region, properties in status]

        return status, elapsed

    def terminate(self):
        """ Terminates the tracker. This method is used to terminate the tracker. It closes the connection to the tracker and terminates the tracker process.
        """
        with self._watchdog_lock:

            if not self.alive:
                return

            if not self._client is None:
                self._client.quit()

            try:
                self._process.wait(3)
            except subprocess.TimeoutExpired:
                pass

            if self._process.returncode is None:
                self._process.terminate()
                try:
                    self._process.wait(3)
                except subprocess.TimeoutExpired:
                    pass

                if self._process.returncode is None:
                    self._process.kill()

            if not self._process.stdout.closed:
                self._process.stdout.close()

            if not self._process.stdin.closed:
                self._process.stdin.close()

            if not self._socket is None:
                self._socket.close()

            self._returncode = self._process.returncode

            self._client = None
            self._process = None

    def __del__(self):
        """ Destructor. This method is used to terminate the tracker process if it is still alive."""
        if hasattr(self, "_workdir"):
            shutil.rmtree(self._workdir, ignore_errors=True)

    def wait(self):
        """ Waits for the tracker to terminate. This method is used to wait for the tracker to terminate. It waits until the tracker process terminates."""

        self._watchdog_reset(True)

        # Flush remaining output
        while True: #self._process.returncode is None:
            line = self._process.stdout.readline()
            if not line is None and not self._client._logger is None:
                self._client._logger.handle(line.decode("utf-8"))

        self._watchdog_reset(False)


class TraxTrackerRuntime(TrackerRuntime):
    """ The TraX tracker runtime. This class is used to run a tracker using the TraX protocol."""

    def __init__(self, tracker: Tracker, command: str, log: bool = False, timeout: int = 30, linkpaths=None, envvars=None, arguments=None, socket=False, restart=False, onerror=None):
        """ Initializes the TraX tracker runtime.

        Args:
            tracker: The tracker to be run.
            command: The command to run the tracker.
            log: Whether to log the output of the tracker.
            timeout: The timeout in seconds for the tracker to respond.
            linkpaths: The paths to be added to the PATH environment variable.
            envvars: The environment variables to be set for the tracker.
            arguments: The arguments to be passed to the tracker.
            socket: Whether to use a socket to communicate with the tracker.
            restart: Whether to restart the tracker if it crashes.
            onerror: The error handler to be called if the tracker crashes.
        """
        super().__init__(tracker)
        self._command = command
        self._process = None
        self._tracker = tracker
        if linkpaths is None:
            linkpaths = []
        if isinstance(linkpaths, str):
            linkpaths = linkpaths.split(os.pathsep)
        linkpaths = normalize_paths(linkpaths, tracker)
        self._socket = to_logical(socket)
        self._restart = to_logical(restart)
        if not log:
            self._output = LogAggregator()
        else:
            self._output = None
        self._timeout = to_number(timeout, min_n=1)
        self._arguments = arguments
        self._onerror = onerror
        self._workdir = None

        if sys.platform.startswith("win"):
            pathvar = "PATH"
        else:
            pathvar = "LD_LIBRARY_PATH"

        envvars[pathvar] = envvars[pathvar] + os.pathsep + os.pathsep.join(linkpaths) if pathvar in envvars else os.pathsep.join(linkpaths)
        envvars["TRAX"] = "1"

        self._envvars = envvars

    @property
    def tracker(self) -> Tracker:
        """ The associated tracker object. """
        return self._tracker

    @property
    def multiobject(self):
        """ Whether the tracker supports multiple objects."""
        self._connect()
        return self._process._multiobject

    def _connect(self):
        """ Connects to the tracker. This method is used to connect to the tracker. It starts the tracker process if it is not running yet."""
        if not self._process:
            if not self._output is None:
                log = self._output
            else:
                log = ColorizedOutput()
            self._process = TrackerProcess(self._command, self._envvars, log=log, socket=self._socket, timeout=self._timeout)
            if self._process.has_vot_wrapper:
                self._restart = True

    def _error(self, exception):
        """ Handles an error. This method is used to handle an error. It calls the error handler if it is set."""
        workdir = None
        timeout = False
        if not self._output is None:
            if not self._process is None:
                if self._process.alive:
                    self._process.terminate()
                
                self._output("Process exited with code ({})\n".format(self._process.returncode))
                timeout = self._process.interrupted
                self._workdir = self._process.workdir
            else:
                self._output("Process not alive anymore, unable to retrieve return code\n")

        log = str(self._output)

        try:

            if not self._onerror is None and isinstance(self._onerror, callable):
                self._onerror(log, workdir)

        except Exception as e:
            logger.exception("Error during error handler for runtime of tracker %s", self._tracker.identifier, exc_info=e)

        if timeout:
            raise TrackerException("Tracker interrupted, it did not reply in {} seconds".format(self._timeout), tracker=self._tracker, \
                tracker_log=log if not self._output is None else None)

        raise TrackerException(exception, tracker=self._tracker, \
            tracker_log=log if not self._output is None else None)

    def restart(self):
        """ Restarts the tracker. This method is used to restart the tracker. It stops the tracker process and starts it again."""
        try:
            self.stop()
            self._connect()
        except TraxException as e:
            self._error(e)

    def initialize(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """ Initializes the tracker. This method is used to initialize the tracker. It starts the tracker process if it is not running yet.
        
        Args:
            frame: The initial frame.
            new: The initial objects.
            properties: The initial properties.
            
        Returns:
            A tuple containing the initial objects and the initial score.
        """
        try:
            if self._restart:
                self.stop()
            self._connect()

            tproperties = dict(self._arguments)

            if not properties is None:
                tproperties.update(properties)

            return self._process.initialize(frame, new, tproperties)
        except TraxException as e:
            self._error(e)

    def update(self, frame: Frame, new: Objects = None, properties: dict = None) -> Tuple[Objects, float]:
        """ Updates the tracker. This method is used to update the tracker state with a new frame.
        
        Args:
            frame: The current frame.
            new: The current objects.
            properties: The current properties.
            
        Returns:
            A tuple containing the updated objects and the updated score.
        """
        try:
            if properties is None:
                properties = dict()
            return self._process.update(frame, new, properties)
        except TraxException as e:
            self._error(e)

    def stop(self):
        """ Stops the tracker. This method is used to stop the tracker. It stops the tracker process."""
        if not self._process is None:
            self._process.terminate()
            self._process = None

    def __del__(self):
        """ Destructor. This method is used to stop the tracker process when the object is deleted."""
        self.stop()

def escape_path(path):
    """ Escapes a path. This method is used to escape a path.
    
    Args:
        path: The path to escape.
        
    Returns:
        The escaped path.
    """
    if sys.platform.startswith("win"):
        return path.replace("\\\\", "\\").replace("\\", "\\\\")
    else:
        return path

def trax_python_adapter(tracker, command, envvars, paths="", log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, python=None, socket=False, restart=False, **kwargs):
    """ Creates a Python adapter for a tracker. This method is used to create a Python adapter for a tracker.

    Args:
        tracker: The tracker to create the adapter for.
        command: The command to run the tracker.
        envvars: The environment variables to set.
        paths: The paths to add to the Python path.
        log: Whether to log the tracker output.
        timeout: The timeout in seconds.
        linkpaths: The paths to link.
        arguments: The arguments to pass to the tracker.
        python: The Python interpreter to use.
        socket: Whether to use a socket to communicate with the tracker.
        restart: Whether to restart the tracker after each frame.
        kwargs: Additional keyword arguments.

    Returns:
        The Python TraX runtime object.
    """
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["sys.path.insert(0, '{}');".format(escape_path(x)) for x in normalize_paths(paths[::-1], tracker)])
    interpreter = sys.executable if python is None else python

    # simple check if the command is only a package name to be imported or a script
    if re.match("^[a-zA-Z_][a-zA-Z0-9_\\.]*$", command) is None:
        # We have to escape all double quotes
        command = command.replace("\"", "\\\"")
        command = '{} -c "import sys;{} {}"'.format(interpreter, pathimport, command)
    else:
        command = '{} -m {}'.format(interpreter, command)

    envvars["PYTHONPATH"] = os.pathsep.join(normalize_paths(paths[::-1], tracker))   
    envvars["PYTHONUNBUFFERED"] = "1"

    return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)

def trax_matlab_adapter(tracker, command, envvars, paths="", log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, matlab=None, socket=False, restart=False, **kwargs):
    """ Creates a Matlab adapter for a tracker. This method is used to create a Matlab adapter for a tracker. 

    Args:
        tracker: The tracker to create the adapter for.
        command: The command to run the tracker.
        envvars: The environment variables to set.
        paths: The paths to add to the Matlab path.
        log: Whether to log the tracker output.
        timeout: The timeout in seconds.
        linkpaths: The paths to link.
        arguments: The arguments to pass to the tracker.
        matlab: The Matlab executable to use.
        socket: Whether to use a socket to communicate with the tracker.
        restart: Whether to restart the tracker after each frame.
        kwargs: Additional keyword arguments.

    Returns:
        The Matlab TraX runtime object.
    """
    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["addpath('{}');".format(x) for x in normalize_paths(paths, tracker)])

    if sys.platform.startswith("win"):
        matlabname = "matlab.exe"
        socket = True # We have to use socket connection in this case
    else:
        matlabname = "matlab"


    if matlab is None:
        matlabroot = os.getenv("MATLAB_ROOT", None)
        if matlabroot is None:
            testdirs = os.getenv("PATH", "").split(os.pathsep)
            for testdir in testdirs:
                if os.path.isfile(os.path.join(testdir, matlabname)):
                    matlabroot = os.path.dirname(testdir)
                    break
            if matlabroot is None:
                raise RuntimeError("Matlab executable not found, set MATLAB_ROOT environmental variable manually.")
        matlab_executable = os.path.join(matlabroot, 'bin', matlabname)
    else:
        matlab_executable = matlab

    if sys.platform.startswith("win"):
        matlab_executable = '"' + matlab_executable + '"'
        matlab_flags = ['-nodesktop', '-nosplash', '-wait', '-minimize']
    else:
        matlab_flags = ['-nodesktop', '-nosplash']

    matlab_script = 'try; diary ''runtime.log''; {}{}; catch ex; disp(getReport(ex)); end; quit;'.format(pathimport, command)

    command = '{} {} -r "{}"'.format(matlab_executable, " ".join(matlab_flags), matlab_script)

    return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)

def trax_octave_adapter(tracker, command, envvars, paths="", log: bool = False, timeout: int = 30, linkpaths=None, arguments=None, socket=False, restart=False, **kwargs):
    """ Creates an Octave adapter for a tracker. This method is used to create an Octave adapter for a tracker. 

    Args:
        tracker: The tracker to create the adapter for.
        command: The command to run the tracker.
        envvars: The environment variables to set.
        paths: The paths to add to the Octave path.
        log: Whether to log the tracker output.
        timeout: The timeout in seconds.
        linkpaths: The paths to link.
        arguments: The arguments to pass to the tracker.
        socket: Whether to use a socket to communicate with the tracker.
        restart: Whether to restart the tracker after each frame.
        kwargs: Additional keyword arguments.

    Returns:
        The Octave TraX runtime object.
    """

    if not isinstance(paths, list):
        paths = paths.split(os.pathsep)

    pathimport = " ".join(["addpath('{}');".format(x) for x in normalize_paths(paths, tracker)])

    octaveroot = os.getenv("OCTAVE_ROOT", None)

    if sys.platform.startswith("win"):
        octavename = "octave.exe"
    else:
        octavename = "octave"

    if octaveroot is None:
        testdirs = os.getenv("PATH", "").split(os.pathsep)
        for testdir in testdirs:
            if os.path.isfile(os.path.join(testdir, octavename)):
                octaveroot = os.path.dirname(testdir)
                break
        if octaveroot is None:
            raise RuntimeError("Octave executable not found, set OCTAVE_ROOT environmental variable manually.")

    if sys.platform.startswith("win"):
        octave_executable = '"' + os.path.join(octaveroot, 'bin', octavename) + '"'
    else:
        octave_executable = os.path.join(octaveroot, 'bin', octavename)

    octave_flags = ['--no-gui', '--no-window-system']

    octave_script = 'try; diary ''runtime.log''; {}{}; catch ex; disp(ex.message); for i = 1:size(ex.stack) disp(''filename''); disp(ex.stack(i).file); disp(''line''); disp(ex.stack(i).line); endfor; end; quit;'.format(pathimport, command)

    command = '{} {} --eval "{}"'.format(octave_executable, " ".join(octave_flags), octave_script)

    return TraxTrackerRuntime(tracker, command, log=log, timeout=timeout, linkpaths=linkpaths, envvars=envvars, arguments=arguments, socket=socket, restart=restart)
