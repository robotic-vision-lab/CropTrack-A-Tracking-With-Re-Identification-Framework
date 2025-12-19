import scipy
import lap
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
import tracker.nsa_kalman_filter as kalman_filter
from tracker.re_ranking import re_ranking


def one_to_many_reid_association(tracks, detections, iou_thresh):
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracks))), tuple(range(len(detections)))

    iou_dist_matrix = iou_distance(tracks, detections)

    potential_matches_indices = np.where(iou_dist_matrix < iou_thresh)

    if potential_matches_indices[0].size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(len(tracks))), tuple(range(len(detections)))

    potential_track_idxs = np.unique(potential_matches_indices[0])
    potential_det_idxs = np.unique(potential_matches_indices[1])

    sub_tracks = [tracks[i] for i in potential_track_idxs]
    sub_detections = [detections[i] for i in potential_det_idxs]

    reid_dist_matrix = re_ranking_distance(sub_tracks, sub_detections)
    reid_dist_matrix = fuse_iou(reid_dist_matrix, sub_tracks, sub_detections)

    match_candidates = []
    for i in range(len(potential_matches_indices[0])):
        track_idx = potential_matches_indices[0][i]
        det_idx = potential_matches_indices[1][i]

        sub_track_idx = np.where(potential_track_idxs == track_idx)[0][0]
        sub_det_idx = np.where(potential_det_idxs == det_idx)[0][0]

        reid_dist = reid_dist_matrix[sub_track_idx, sub_det_idx]
        if reid_dist > 0.5:
            continue
        match_candidates.append((reid_dist, track_idx, det_idx))

    match_candidates.sort()

    matches = []
    matched_tracks = set()
    matched_dets = set()

    for reid_dist, track_idx, det_idx in match_candidates:
        if track_idx not in matched_tracks and det_idx not in matched_dets:
            matches.append([track_idx, det_idx])
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)

    all_track_indices = set(range(len(tracks)))
    all_det_indices = set(range(len(detections)))

    unmatched_a = tuple(all_track_indices - matched_tracks)
    unmatched_b = tuple(all_det_indices - matched_dets)

    return np.asarray(matches), unmatched_a, unmatched_b

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.tlbr) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.tlbr) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def bbox_center_distance(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        axy = [track.to_xyah()[:2] for track in atracks]
        bxy = [track.to_xyah()[:2] for track in btracks]

    A = np.asarray(axy)
    B = np.asarray(bxy)
    A_norm = np.sum(A ** 2, axis=1).reshape(-1, 1)  # (m, 1)
    B_norm = np.sum(B ** 2, axis=1).reshape(1, -1)  # (1, n)

    # Squared Euclidean distance
    D = A_norm + B_norm - 2 * A @ B.T

    # Clip tiny negatives (numerical stability)
    D = np.maximum(D, 0.0)
    return  np.sqrt(D)

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    #track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    track_features = np.asarray([track.curr_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def re_ranking_distance(tracks, detections, nei_rad=600):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    t_idx = []
    track_features = []
    for i, t in enumerate(tracks):
        track_features.extend(t.features)
        t_idx.extend([i]*len(t.features))

    track_features = np.vstack(track_features)
    dist_matrix = re_ranking(det_features, track_features) # return num_det * num_track

    m, n = dist_matrix.shape
    unique_classes, inverse_idx = np.unique(t_idx, return_inverse=True)
    num_classes = unique_classes.shape[0]

    # Initialize with +inf
    reduced = np.full((m, num_classes), np.inf, dtype=dist_matrix.dtype)

    # track wise min dist, each track could have multiple features
    # For each row, scatter minimum across classes
    for i in range(m):
        np.minimum.at(reduced[i], inverse_idx, dist_matrix[i])

    center_dis = bbox_center_distance(tracks, detections)
    neighbor_mask =  np.where(center_dis < nei_rad, 1, np.inf)
    dis  = reduced.T * neighbor_mask
    return dis

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = 0.75*reid_sim + (0.25*iou_sim)
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_iou_embedding(tracks, detections, metric='cosine'):
    cost_matrix = embedding_distance(tracks, detections, metric)
    #return cost_matrix
    alpha = .7

    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = np.where(  iou_sim>0., reid_sim, 0)
    # fuse_sim =  reid_sim * (1 + iou_sim) / 2
    # det_scores = np.array([det.score for det in detections])
    # det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost
