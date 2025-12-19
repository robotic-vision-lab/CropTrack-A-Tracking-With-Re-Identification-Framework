import os
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm
from datasets import grapes_dataset
from datasets import cotton_dataset
from tracker.croptrack import CropTrack

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def make_parser():
    parser = argparse.ArgumentParser("CropTrack Eval")  # Renamed for clarity
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # Dataset args
    parser.add_argument("--data_dir", type=str, required=True, help="absolute path to the dataset directory")
    parser.add_argument("--dataset_type", type=str, default="cotton", choices=["cotton", "grape"],
                        help="choose dataset loader")

    # Tracking args
    parser.add_argument("--alpha", type=list, default=[0.5, 0.9, 0.1], help="alpha values for EMA, None for no EMA")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.875, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')

    # Det args (kept for compatibility)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_false", help="test mot20.")

    return parser


args = make_parser().parse_args()

# Select dataset based on argument
if args.dataset_type == "cotton":
    track_seq = cotton_dataset.CottonDataset(args.data_dir, 'test')
elif args.dataset_type == "grape":
    track_seq = grapes_dataset.GrapeDataset(args.data_dir, 'test')
else:
    raise ValueError(f"Unknown dataset type: {args.dataset_type}")


def update_metric(metric_acc, gt_id, gt_box, match_id, match_bb, w_margin):
    dis_mat = mm.distances.iou_matrix(gt_box, match_bb, max_iou=0.65)
    metric_acc.update(gt_id, match_id, dis_mat)


def populate_result(seq_name, metric_acc):
    mh = mm.metrics.create()
    summary = mh.compute(metric_acc, metrics=mm.metrics.motchallenge_metrics,
                         name=seq_name)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print('\n', strsummary)
    return summary


summary_df = []

mot_output_dir = "results"
os.makedirs(mot_output_dir, exist_ok=True)

# Main loop
for n, seq in enumerate(track_seq):
    tracker = CropTrack(args)
    metric_acc = mm.MOTAccumulator(auto_id=True)

    mot_output_file = os.path.join(mot_output_dir, f"{seq.seq_name}.txt")
    with open(mot_output_file, "w") as f_out:

        for frame_idx in range(len(seq.data)):
            frame_data = seq[frame_idx]
            dets = frame_data['dets']
            feats = frame_data['feats']
            gt = frame_data['gt']
            img_path = frame_data['img_path']

            dets[:, 2:4] += dets[:, 0:2]
            gt_bb_id = list(gt.keys())
            gt_bb = np.asarray([gt[k] for k in gt_bb_id])

            online_targets, vis_data = tracker.update(dets, img_path, [1, 1], [1, 1], feats)

            bb = []
            bb_id = []
            for t in online_targets:
                tlwh = t.tlwh
                track_id = t.track_id

                f_out.write(
                    f"{frame_idx + 1},{track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1,-1,-1,-1\n"
                )

                bb.append(tlwh.tolist())
                bb_id.append(track_id)

            update_metric(metric_acc, gt_bb_id, gt_bb, bb_id, bb, 0)

    summary = populate_result(seq.seq_name, metric_acc)
    summary_df.append(summary)

summary_df = pd.concat(summary_df, ignore_index=True)