"""This module contains the implementation of the long term tracking performance measures."""
import math
import numpy as np
from typing import List, Iterable, Tuple, Any
import itertools

from attributee import Float, Integer, Boolean, Include

from vot.tracker import Tracker
from vot.dataset import Sequence
from vot.region import Region, RegionType, calculate_overlaps
from vot.experiment import Experiment
from vot.experiment.multirun import UnsupervisedExperiment, MultiRunExperiment
from vot.analysis import SequenceAggregator, Analysis, SeparableAnalysis, \
    MissingResultsException, Measure, Sorting, Curve, Plot, SequenceAggregator, \
    Axes, analysis_registry, Point, is_special, Analysis
from vot.utilities.data import Grid

def determine_thresholds(scores: Iterable[float], resolution: int) -> List[float]:
    """Determine thresholds for a given set of scores and a resolution. 
    The thresholds are determined by sorting the scores and selecting the thresholds that divide the sorted scores into equal sized bins. 
    
    Args:
        scores (Iterable[float]): Scores to determine thresholds for.
        resolution (int): Number of thresholds to determine.
        
    Returns:
        List[float]: List of thresholds.
    """
    scores = [score for score in scores if not math.isnan(score)] #and not score is None]
    scores = sorted(scores, reverse=True)

    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        idxs = np.round(np.linspace(delta, len(scores) - delta, num=resolution - 2)).astype(np.int)
        thresholds = [scores[idx] for idx in idxs]
    else:
        thresholds = scores

    thresholds.insert(0, math.inf)
    thresholds.insert(len(thresholds), -math.inf)

    return thresholds

def compute_tpr_curves(trajectory: List[Region], confidence: List[float], sequence: Sequence, thresholds: List[float],
    ignore_unknown: bool = True, bounded: bool = True):
    """Compute the TPR curves for a given trajectory and confidence scores. 
    
    Args:
        trajectory (List[Region]): Trajectory to compute the TPR curves for.
        confidence (List[float]): Confidence scores for the trajectory.
        sequence (Sequence): Sequence to compute the TPR curves for.
        thresholds (List[float]): Thresholds to compute the TPR curves for.
        ignore_unknown (bool, optional): Ignore unknown regions. Defaults to True.
        bounded (bool, optional): Bounded evaluation. Defaults to True.

    Returns:
        List[float], List[float]: TPR curves for the given thresholds.
    """

    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    confidence = np.array(confidence)

    n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])

    precision = len(thresholds) * [float(0)]
    recall = len(thresholds) * [float(0)]

    for i, threshold in enumerate(thresholds):

        subset = confidence >= threshold

        if np.sum(subset) == 0:
            precision[i] = 1
            recall[i] = 0
        else:
            precision[i] = np.mean(overlaps[subset])
            recall[i] = np.sum(overlaps[subset]) / n_visible

    return precision, recall

class _ConfidenceScores(SeparableAnalysis):
    """Computes the confidence scores for a tracker for given sequences. This is internal analysis and should not be used directly."""

    @property
    def _title_default(self):
        """Title of the analysis."""
        return "Aggregate confidence scores"

    def describe(self):
        """Describes the analysis."""
        return None,

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. """
        return isinstance(experiment, UnsupervisedExperiment)

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Computes the confidence scores for a tracker for given sequences.
        
        Args:
            experiment (Experiment): Experiment to compute the confidence scores for.
            tracker (Tracker): Tracker to compute the confidence scores for.
            sequence (Sequence): Sequence to compute the confidence scores for.
            dependencies (List[Grid]): Dependencies of the analysis.
            
        Returns:
            Tuple[Any]: Confidence scores for the given sequence.
        """

        scores_all = []
        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException("Missing results for sequence {}".format(sequence.name))

        for trajectory in trajectories:
            confidence = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
            scores_all.extend(confidence)

        return scores_all,

class _Thresholds(SequenceAggregator):
    """Computes the thresholds for a tracker for given sequences. This is internal analysis and should not be used directly."""

    resolution = Integer(default=100)

    @property
    def _title_default(self):
        """Title of the analysis."""
        return "Thresholds for tracking precision/recall"

    def describe(self):
        """Describes the analysis."""
        return None,

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. """
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        """Dependencies of the analysis."""
        return _ConfidenceScores(),

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:
        """Computes the thresholds for a tracker for given sequences.
        
        Args:    
            tracker (Tracker): Tracker to compute the thresholds for.
            sequences (List[Sequence]): Sequences to compute the thresholds for.
            results (Grid): Results of the dependencies.
            
        Returns:
            Tuple[Any]: Thresholds for the given sequences."""

        thresholds = determine_thresholds(itertools.chain(*[result[0] for result in results]), self.resolution),

        return thresholds,

@analysis_registry.register("pr_curves")
class PrecisionRecallCurves(SeparableAnalysis):
    """ Computes the precision/recall curves for a tracker for given sequences. """

    thresholds = Include(_Thresholds)
    ignore_unknown = Boolean(default=True, description="Ignore unknown regions")
    bounded = Boolean(default=True, description="Bounded evaluation")

    @property
    def _title_default(self):
        """Title of the analysis."""
        return "Tracking precision/recall"

    def describe(self):
        """Describes the analysis."""
        return Curve("Precision Recall curve", dimensions=2, abbreviation="PR", minimal=(0, 0), maximal=(1, 1), labels=("Recall", "Precision")), None

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis."""
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        """Dependencies of the analysis."""
        return self.thresholds,

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Computes the precision/recall curves for a tracker for given sequences. 
        
        Args:
            experiment (Experiment): Experiment to compute the precision/recall curves for.
            tracker (Tracker): Tracker to compute the precision/recall curves for.
            sequence (Sequence): Sequence to compute the precision/recall curves for.
            dependencies (List[Grid]): Dependencies of the analysis.
            
        Returns:
            Tuple[Any]: Precision/recall curves for the given sequence.
        """

        thresholds = dependencies[0, 0][0][0] # dependencies[0][0, 0]

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        precision = len(thresholds) * [float(0)]
        recall = len(thresholds) * [float(0)]
        for trajectory in trajectories:
            confidence = [trajectory.properties(i).get('confidence', 0) for i in range(len(trajectory))]
            pr, re = compute_tpr_curves(trajectory.regions(), confidence, sequence, thresholds, self.ignore_unknown, self.bounded)
            for i in range(len(thresholds)):
                precision[i] += pr[i]
                recall[i] += re[i]

#         return [(re / len(trajectories), pr / len(trajectories)) for pr, re in zip(precision, recall)], thresholds
        return [(pr / len(trajectories), re / len(trajectories)) for pr, re in zip(precision, recall)], thresholds

@analysis_registry.register("pr_curve")
class PrecisionRecallCurve(SequenceAggregator):
    """ Computes the average precision/recall curve for a tracker. """

    curves = Include(PrecisionRecallCurves)

    @property
    def _title_default(self):
        """Title of the analysis."""
        return "Tracking precision/recall average curve"

    def describe(self):
        """Describes the analysis."""
        return self.curves.describe()

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. This analysis is compatible with unsupervised experiments."""
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        """Dependencies of the analysis."""
        return self.curves,

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid) -> Tuple[Any]:
        """Computes the average precision/recall curve for a tracker. 
        
        Args:
            tracker (Tracker): Tracker to compute the average precision/recall curve for.
            sequences (List[Sequence]): Sequences to compute the average precision/recall curve for.
            results (Grid): Results of the dependencies.
            
        Returns:
            Tuple[Any]: Average precision/recall curve for the given sequences.
        """

        curve = None
        thresholds = None

        for partial, thresholds in results:
            if curve is None:
                curve = partial
                continue

            curve = [(pr1 + pr2, re1 + re2) for (pr1, re1), (pr2, re2) in zip(curve, partial)]

        curve = [(re / len(results), pr / len(results)) for pr, re in curve]

        return curve, thresholds


@analysis_registry.register("f_curve")
class FScoreCurve(Analysis):
    """ Computes the F-score curve for a tracker. """

    beta = Float(default=1, description="Beta value for the F-score")
    prcurve = Include(PrecisionRecallCurve)

    @property
    def _title_default(self):
        """Title of the analysis."""
        return "Tracking precision/recall"

    def describe(self):
        """Describes the analysis."""
        return Plot("Tracking F-score curve", "F", wrt="normalized threshold", minimal=0, maximal=1), None

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. This analysis is compatible with unsupervised experiments."""
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        """Dependencies of the analysis."""
        return self.prcurve,

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        """Computes the F-score curve for a tracker. 
        
        Args:
            experiment (Experiment): Experiment to compute the F-score curve for.
            trackers (List[Tracker]): Trackers to compute the F-score curve for.
            sequences (List[Sequence]): Sequences to compute the F-score curve for.
            dependencies (List[Grid]): Dependencies of the analysis.
            
        Returns:
            Grid: F-score curve for the given sequences.
        """

        processed_results = Grid(len(trackers), 1)

        for i, result in enumerate(dependencies[0]):
            beta2 = (self.beta * self.beta)
            f_curve = [((1 + beta2) * pr_ * re_) / (beta2 * pr_ + re_) for pr_, re_ in result[0]]

            processed_results[i, 0] = (f_curve, result[0][1])

        return processed_results

    @property
    def axes(self):
        """Axes of the analysis."""
        return Axes.TRACKERS

@analysis_registry.register("average_tpr")
class PrecisionRecall(Analysis):
    """ Computes the average precision/recall for a tracker. """

    prcurve = Include(PrecisionRecallCurve)
    fcurve = Include(FScoreCurve)

    @property
    def _title_default(self):
        """Title of the analysis."""
        return "Tracking precision/recall"

    def describe(self):
        """Describes the analysis."""
        return Measure("Precision", "Pr", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Recall", "Re", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("F Score", "F", minimal=0, maximal=1, direction=Sorting.DESCENDING)

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. This analysis is compatible with unsupervised experiments."""
        return isinstance(experiment, UnsupervisedExperiment)

    def dependencies(self):
        """Dependencies of the analysis."""
        return self.prcurve, self.fcurve

    def compute(self, experiment: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        """Computes the average precision/recall for a tracker. 
        
        Args:
            experiment (Experiment): Experiment to compute the average precision/recall for.
            trackers (List[Tracker]): Trackers to compute the average precision/recall for.
            sequences (List[Sequence]): Sequences to compute the average precision/recall for.
            dependencies (List[Grid]): Dependencies of the analysis.

        Returns:
            Grid: Average precision/recall for the given sequences.
        """

        f_curves = dependencies[1]
        pr_curves = dependencies[0]

        joined = Grid(len(trackers), 1)

        for i, (f_curve, pr_curve) in enumerate(zip(f_curves, pr_curves)):
            # get optimal F-score and Pr and Re at this threshold
            f_score = max(f_curve[0])
            best_i = f_curve[0].index(f_score)
            re_score = pr_curve[0][best_i][0]
            pr_score = pr_curve[0][best_i][1]
            joined[i, 0] = (pr_score, re_score, f_score)

        return joined

    @property
    def axes(self):
        """Axes of the analysis."""
        return Axes.TRACKERS


def count_frames(trajectory: List[Region], groundtruth: List[Region], bounds = None, threshold: float = 0) -> float:
    """Counts the number of frames where the tracker is correct, fails, misses, hallucinates or notices an object.
    
    Args:
        trajectory (List[Region]): Trajectory of the tracker.
        groundtruth (List[Region]): Groundtruth trajectory.
        bounds (Optional[Region]): Bounds of the sequence.
        threshold (float): Threshold for the overlap.
        
    Returns:
        float: Number of frames where the tracker is correct, fails, misses, hallucinates or notices an object.
    """

    overlaps = np.array(calculate_overlaps(trajectory, groundtruth, bounds))
    if threshold is None: threshold = -1

    # Tracking, Failure, Miss, Halucination, Notice
    T, F, M, H, N = 0, 0, 0, 0, 0

    for i, (region_tr, region_gt) in enumerate(zip(trajectory, groundtruth)):
        if (is_special(region_gt, Sequence.UNKNOWN)):
            continue
        if region_gt.is_empty():
            if region_tr.is_empty():
                N += 1
            else:
                H += 1
        else:
            if overlaps[i] > threshold:
                T += 1
            else:
                if region_tr.is_empty():
                    M += 1
                else:
                    F += 1

    return T, F, M, H, N

class CountFrames(SeparableAnalysis):
    """Counts the number of frames where the tracker is correct, fails, misses, hallucinates or notices an object."""

    threshold = Float(default=0.0, val_min=0, val_max=1)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. This analysis is compatible with multi-run experiments."""
        return isinstance(experiment, MultiRunExperiment)

    def describe(self):
        """Describes the analysis."""
        return None, 

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Computes the number of frames where the tracker is correct, fails, misses, hallucinates or notices an object."""

        assert isinstance(experiment, MultiRunExperiment)

        objects = sequence.objects()
        distribution = []
        bounds = (sequence.size) if self.bounded else None

        for object in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[object])
            if len(trajectories) == 0:
                raise MissingResultsException()

            CN, CF, CM, CH, CT = 0, 0, 0, 0, 0

            for trajectory in trajectories:
                T, F, M, H, N = count_frames(trajectory.regions(), sequence.object(object), bounds=bounds)
                CN += N
                CF += F
                CM += M
                CH += H
                CT += T
            CN /= len(trajectories)
            CF /= len(trajectories)
            CM /= len(trajectories)
            CH /= len(trajectories)
            CT /= len(trajectories)

            distribution.append((CT, CF, CM, CH, CN))

        return distribution,


@analysis_registry.register("quality_auxiliary")
class QualityAuxiliary(SeparableAnalysis):
    """Computes the non-reported error, drift-rate error and absence-detection quality."""

    threshold = Float(default=0.0, val_min=0, val_max=1)
    bounded = Boolean(default=True)
    absence_threshold = Integer(default=10, val_min=0)

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. This analysis is compatible with multi-run experiments."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self):
        """Default title of the analysis."""
        return "Quality Auxiliary"

    def describe(self):
        """Describes the analysis."""
        return Measure("Non-reported Error", "NRE", 0, 1, Sorting.DESCENDING), \
            Measure("Drift-rate Error", "DRE", 0, 1, Sorting.DESCENDING), \
            Measure("Absence-detection Quality", "ADQ", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:
        """Computes the non-reported error, drift-rate error and absence-detection quality.
        
        Args:
            experiment (Experiment): Experiment.
            tracker (Tracker): Tracker.
            sequence (Sequence): Sequence.
            dependencies (List[Grid]): Dependencies.
            
        Returns:
            Tuple[Any]: Non-reported error, drift-rate error and absence-detection quality.

        """

        assert isinstance(experiment, MultiRunExperiment)

        not_reported_error = 0
        drift_rate_error = 0
        absence_detection = 0

        objects = sequence.objects()
        bounds = (sequence.size) if self.bounded else None

        absence_valid = 0

        for object in objects:
            trajectories = experiment.gather(tracker, sequence, objects=[object])
            if len(trajectories) == 0:
                raise MissingResultsException()

            CN, CF, CM, CH, CT = 0, 0, 0, 0, 0

            for trajectory in trajectories:
                T, F, M, H, N = count_frames(trajectory.regions(), sequence.object(object), bounds=bounds)
                CN += N
                CF += F
                CM += M
                CH += H
                CT += T
            CN /= len(trajectories)
            CF /= len(trajectories)
            CM /= len(trajectories)
            CH /= len(trajectories)
            CT /= len(trajectories)

            not_reported_error += CM / (CT + CF + CM)
            drift_rate_error += CF / (CT + CF + CM)

            if CN + CH > self.absence_threshold:
                absence_detection += CN / (CN + CH)
                absence_valid += 1

        if absence_valid > 0:
            absence_detection /= absence_valid
        else:
            absence_detection = None

        return not_reported_error / len(objects), drift_rate_error / len(objects), absence_detection,


@analysis_registry.register("average_quality_auxiliary")
class AverageQualityAuxiliary(SequenceAggregator):
    """Computes the average non-reported error, drift-rate error and absence-detection quality."""

    analysis = Include(QualityAuxiliary)

    @property
    def _title_default(self):
        """Default title of the analysis."""
        return "Quality Auxiliary"

    def dependencies(self):
        """Returns the dependencies of the analysis."""
        return self.analysis,

    def describe(self):
        """Describes the analysis."""
        return Measure("Non-reported Error", "NRE", 0, 1, Sorting.DESCENDING), \
            Measure("Drift-rate Error", "DRE", 0, 1, Sorting.DESCENDING), \
            Measure("Absence-detection Quality", "ADQ", 0, 1, Sorting.DESCENDING),

    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. This analysis is compatible with multi-run experiments."""
        return isinstance(experiment, MultiRunExperiment)

    def aggregate(self, tracker: Tracker, sequences: List[Sequence], results: Grid):
        """Aggregates the non-reported error, drift-rate error and absence-detection quality.
        
        Args:
            tracker (Tracker): Tracker.
            sequences (List[Sequence]): Sequences.
            results (Grid): Results.
            
        Returns:
            Tuple[Any]: Non-reported error, drift-rate error and absence-detection quality.
        """

        not_reported_error = 0
        drift_rate_error = 0
        absence_detection = 0
        absence_count = 0

        for nre, dre, ad in results:
            not_reported_error += nre
            drift_rate_error += dre
            if ad is not None:
                absence_count += 1
                absence_detection += ad

        if absence_count > 0:
            absence_detection /= absence_count

        return not_reported_error / len(sequences), drift_rate_error / len(sequences), absence_detection

from vot.analysis import SequenceAggregator
from vot.analysis.accuracy import SequenceAccuracy

@analysis_registry.register("longterm_ar")
class AccuracyRobustness(Analysis):
    """Longterm multi-object accuracy-robustness measure. """

    threshold = Float(default=0.0, val_min=0, val_max=1)
    bounded = Boolean(default=True)
    counts = Include(CountFrames)

    def dependencies(self) -> List[Analysis]:
        """Returns the dependencies of the analysis."""
        return self.counts, SequenceAccuracy(burnin=0, threshold=self.threshold, bounded=self.bounded, ignore_invisible=True, ignore_unknown=False)
    
    def compatible(self, experiment: Experiment):
        """Checks if the experiment is compatible with the analysis. This analysis is compatible with multi-run experiments."""
        return isinstance(experiment, MultiRunExperiment)

    @property
    def _title_default(self):
        """Default title of the analysis."""
        return "Accuracy-robustness"

    def describe(self):
        """Describes the analysis."""
        return Measure("Accuracy", "A", minimal=0, maximal=1, direction=Sorting.DESCENDING), \
             Measure("Robustness", "R", minimal=0, direction=Sorting.DESCENDING), \
             Point("AR plot", dimensions=2, abbreviation="AR", minimal=(0, 0), \
                maximal=(1, 1), labels=("Robustness", "Accuracy"), trait="ar")

    def compute(self, _: Experiment, trackers: List[Tracker], sequences: List[Sequence], dependencies: List[Grid]) -> Grid:
        """Aggregate results from multiple sequences into a single value.
        
        Args:
            experiment (Experiment): Experiment.
            trackers (List[Tracker]): Trackers.
            sequences (List[Sequence]): Sequences.
            dependencies (List[Grid]): Dependencies.
            
        Returns:
            Grid: Aggregated results.
        """

        frame_counts = dependencies[0]
        accuracy_analysis = dependencies[1]

        results = Grid(len(trackers), 1)

        for j, _ in enumerate(trackers):
            accuracy = 0
            robustness = 0
            count = 0

            for i, _ in enumerate(sequences):
                if accuracy_analysis[j, i] is None:
                    continue

                accuracy += accuracy_analysis[j, i][0]

                frame_counts_sequence = frame_counts[j, i][0]

                objects = len(frame_counts_sequence)
                for o in range(objects):
                    robustness += (1/objects) * frame_counts_sequence[o][0] / (frame_counts_sequence[o][0] + frame_counts_sequence[o][1] + frame_counts_sequence[o][2])

                count += 1

            results[j, 0] = (accuracy / count, robustness / count, (robustness / count, accuracy / count))

        return results

    @property
    def axes(self) -> Axes:
        """Returns the axes of the analysis."""
        return Axes.TRACKERS