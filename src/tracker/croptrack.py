import numpy as np
import tracker.matching as matching
from tracker.nsa_kalman_filter import KalmanFilter
from tracker.matching import one_to_many_reid_association, fuse_iou
from tracker.basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feature=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.curr_feat = feature
        self.features = []

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, alpha=[0.5, 0.9, 0.1]):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        if alpha is not None:
            if len(self.features) < len(alpha):
                self.features = [self.curr_feat] * len(alpha)
            else:
                weights = alpha
                for i, w in enumerate(weights):
                    self.features[i] = w * self.curr_feat + (1 - w) * self.features[i]
        else:
            self.features.append(self.curr_feat)

    def re_activate(self, new_track, frame_id, new_id=False, new_feat=None, alpha=[0.5, 0.9, 0.1]):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        weights = alpha
        if alpha is not None and len(self.features) == len(weights):
            for i, w in enumerate(weights):
                self.features[i] = w * new_feat + (1 - w) * self.features[i]
        else:
            self.features.append(new_feat)

    def update(self, new_track, frame_id, new_feat=None, alpha=[0.5, 0.9, 0.1]):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.curr_feat = new_feat
        weights = alpha
        if alpha is not None and len(self.features) == len(weights):
            for i, w in enumerate(weights):
                self.features[i] = w * new_feat + (1 - w) * self.features[i]
        else:
            self.features.append(new_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class CropTrack(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_path, img_info, img_size, feats=None, online_dataset=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        USE_EMBEDDING = True

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        feats_keep = feats[remain_inds]
        feats_second = feats[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                          (tlbr, s, f) in zip(dets, scores_keep, feats_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        STrack.multi_predict(strack_pool)

        ''' Step 2: First association, with high score detection boxes'''
        matches, u_track, u_detection = one_to_many_reid_association(
            strack_pool, detections, iou_thresh=self.args.match_thresh
        )

        first_match_detection = {}
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            first_match_detection[track.track_id] = det

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, det.curr_feat, self.args.alpha)  ################ push feature
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False,
                                  new_feat=det.curr_feat, alpha=self.args.alpha)  ############### push feature
                refind_stracks.append(track)

        '''Cascaded feature-based association - remaining tracks with remaining high score detection boxes'''
        sim_based_association = {}

        sim_candidate_tracks = [strack_pool[i] for i in u_track ]
        sim_candidate_dets = [detections[i] for i in u_detection]
        dists = matching.re_ranking_distance(sim_candidate_tracks, sim_candidate_dets, nei_rad=300) #300
        dists = matching.fuse_score(dists, sim_candidate_dets)
        dists = fuse_iou(dists, sim_candidate_tracks, sim_candidate_dets)

        matches, ux_track, ux_detection = matching.linear_assignment(dists, thresh=0.5)

        u_track = [u_track[ux] for ux in ux_track ]
        u_detection = [u_detection[ux] for ux in ux_detection]
        for itracked, idet in matches:
            track = sim_candidate_tracks[itracked]
            det = sim_candidate_dets[idet]
            sim_based_association[track.track_id] = det

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, det.curr_feat)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False,
                                  new_feat=det.curr_feat)
                refind_stracks.append(track)


        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s,  f) for
                          (tlbr, s, f) in zip(dets_second, scores_second, feats_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)

        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.6)
        second_match_detection = {}
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            second_match_detection[track.track_id] = det

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, det.curr_feat)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, new_feat= det.curr_feat)
                refind_stracks.append(track)


        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]

        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        third_match_detection = {}
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.6) #0.7
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, detections[idet].curr_feat)
            activated_starcks.append(unconfirmed[itracked])
            third_match_detection[unconfirmed[itracked].track_id] = detections[idet]
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]



        return output_stracks, {'strack_pool':strack_pool,
                                'first_match_detection': first_match_detection,
                                'second_match_detection':second_match_detection,
                                'third_match_detection': third_match_detection,
                                'unmatched_det': {-i:detections[i] for i in u_detection}}


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
