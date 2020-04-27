import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
import json


def _distance(x, y):
    return np.linalg.norm(x - y, axis=-1)


def _ground_truth_to_prediction_distance(pred, gt):
    tree = cKDTree(pred)
    dist, _ = tree.query(gt)
    return dist


def _compute_stats(x):
    tmp = {
        'mean': np.mean(x),
        'mse': np.mean(x**2),
        'var': np.var(x),
        'min': np.min(x),
        'max': np.max(x),
        'median': np.median(x),
    }
    tmp = {k: float(v) for k, v in tmp.items()}
    tmp['num_particles'] = x.shape[0]
    return tmp


class FluidErrors:

    def __init__(self):
        self.errors = {}

    def add_errors(self,
                   scene,
                   initialization_frame,
                   current_frame,
                   predicted_pos,
                   gt_pos,
                   compute_gt2pred_distance=False):
        """
        scene: str identifying the scene or sequence
        initialization_frame: frame index that has been used for initialization
        current_frame: frame index of the predicted positions
        predicted_pos: prediction
        gt_pos: corresponding gt positions
        """
        if not initialization_frame < current_frame:
            raise ValueError(
                "initialization_frame {} must be smaller than current_frame {}".
                format(initialization_frame, current_frame))
        item_id = (str(scene), int(initialization_frame), int(current_frame))
        if np.count_nonzero(~np.isfinite(predicted_pos)):
            print('predicted_pos contains nonfinite values')
            return

        if np.count_nonzero(~np.isfinite(gt_pos)):
            print('gt_pos contains nonfinite values')
            return

        errs = _compute_stats(_distance(predicted_pos, gt_pos))

        if compute_gt2pred_distance:
            gt_to_pred_distances = _ground_truth_to_prediction_distance(
                predicted_pos, gt_pos)
            gt_to_pred_errs = _compute_stats(gt_to_pred_distances)
            for k, v in gt_to_pred_errs.items():
                errs['gt2pred_' + k] = v

        if not item_id in self.errors:
            self.errors[item_id] = errs
        else:
            self.errors[item_id].update(errs)

    def get_keys(self):
        scene_ids = set()
        init_frames = set()
        current_frames = set()
        for scene_id, init_frame, current_frame in self.errors:
            scene_ids.add(scene_id)
            init_frames.add(init_frame)
            current_frames.add(current_frame)
        return sorted(scene_ids), sorted(init_frames), sorted(current_frames)

    def save(self, path):
        with open(path, 'w') as f:
            tmp = list(self.errors.items())
            json.dump(tmp, f, indent=4)

    def load(self, path):
        with open(path, 'r') as f:
            tmp = json.load(f)
            self.errors = {tuple(k): v for k, v in tmp}
