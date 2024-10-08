import pickle
import typing as t

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter


def get_frames(video_path: str):
    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame
    video.release()


def count_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def render(data_root_dir: str):
    video_path = f'{data_root_dir}/video/uploaded.mp4'
    csv_path = f'{data_root_dir}/output/uploaded_stride.csv'
    raw_csv_path = f'{data_root_dir}/csv/uploaded.csv'
    keypoint_path = f'{data_root_dir}/output/2d/uploaded.mp4.npz'
    tt_pickle_path = f'{data_root_dir}/output/tt.pickle'
    output_video_path = f'{data_root_dir}/output/render.mp4'

    with open(tt_pickle_path, 'rb') as handle:
        raw_tt = pickle.load(handle)

    df = pd.read_csv(csv_path, header=0)  # noqa
    raw_df = pd.read_csv(raw_csv_path, names=["time", "left.y", "left.x", "left.dt", "right.y", "right.x", "right.dt"])  # noqa

    kepoints = np.load(keypoint_path, allow_pickle=True)

    frames = []
    for frame in get_frames(video_path):
        frames.append(frame)

    # frame_id = 300

    image_height, image_width, _ = frames[0].shape
    dpi = 300
    writer = FFMpegWriter(fps=30)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(image_width / dpi, image_height / dpi))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    n = len(kepoints.f.keypoints)
    with writer.saving(fig, output_video_path, dpi=dpi):
        for frame_id in range(n):
            ax.clear()

            ax.imshow(frames[frame_id][:, :, ::-1])
            xx = kepoints.f.keypoints[frame_id][1].reshape(-1, 17)[0, :]
            yy = kepoints.f.keypoints[frame_id][1].reshape(-1, 17)[1, :]
            ax.scatter(
                x=xx,
                y=yy,
                s=15,
                color='crimson')
            ax.plot(
                [xx[10], xx[8], xx[6], xx[5], xx[7], xx[9]], [yy[10], yy[8], yy[6], yy[5], yy[7], yy[9]],  # noqa
                color='crimson',
            )
            ax.plot([xx[6], xx[12], xx[14], xx[16]], [yy[6], yy[12], yy[14], yy[16]], color='crimson')  # noqa
            ax.plot([xx[5], xx[11], xx[13], xx[15]], [yy[5], yy[11], yy[13], yy[15]], color='crimson')  # noqa
            ax.plot([xx[12], xx[11]], [yy[12], yy[11]], color='crimson')

            # Annotate the current frame type
            # current_frame_type = segmentations[frame_id]
            # if current_frame_type == '-':
            #     current_frame_type = 'none'
            current_frame_type = 'walking'
            if raw_tt[frame_id] == 1:
                current_frame_type = 'turning'
            ax.annotate(
                current_frame_type,
                (100, 200),
                color='white',
                bbox=dict(facecolor='crimson', edgecolor='black'),
            )

            ax.axis('off')

            writer.grab_frame()


def gen_pairs(keypoint_idx_list: t.List[int]):
    pairs = [(keypoint_idx_list[i], keypoint_idx_list[i + 1]) for i in range(len(keypoint_idx_list) - 1)]  # noqa
    return pairs


def new_render(
    video_path: str,
    detectron_custom_dataset_path: str,  # custom data
    tt_pickle_path: str,
    output_video_path: str,
    draw_keypoint: bool = False,
    draw_background: bool = True,
) -> None:

    with open(tt_pickle_path, 'rb') as handle:
        raw_tt = pickle.load(handle)

    detectron_custom_dataset = np.load(detectron_custom_dataset_path, allow_pickle=True)
    keys = list(detectron_custom_dataset.f.positions_2d.item().keys())
    if len(keys) != 1:
        raise ValueError(f'Custom dataset has multiple keys: {keys}')
    key = keys[0]
    keypoints = detectron_custom_dataset.f.positions_2d.item()[key]['custom'][0]

    frames = []
    for frame in get_frames(video_path):
        if draw_background:
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frame))

    image_height, image_width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (image_width, image_height))

    n = len(keypoints)
    for frame_id in range(n):
        frame = frames[frame_id]
        if draw_keypoint:
            color = (0, 0, 255)  # red
            if not draw_background:
                color = (255, 255, 255)

            for point in keypoints[frame_id]:
                cv2.circle(frame, (int(point[0]), int(point[1])), 10, color, -1)

            for (from_idx, to_idx) in gen_pairs([10, 8, 6, 5, 7, 9]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx].astype(int)),
                    tuple(keypoints[frame_id][to_idx].astype(int)),
                    color,
                    5,
                )

            for (from_idx, to_idx) in gen_pairs([6, 12, 14, 16]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx].astype(int)),
                    tuple(keypoints[frame_id][to_idx].astype(int)),
                    color,
                    5,
                )

            for (from_idx, to_idx) in gen_pairs([5, 11, 13, 15]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx].astype(int)),
                    tuple(keypoints[frame_id][to_idx].astype(int)),
                    color,
                    5,
                )

            for (from_idx, to_idx) in gen_pairs([12, 11]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx].astype(int)),
                    tuple(keypoints[frame_id][to_idx].astype(int)),
                    color,
                    5,
                )

        current_frame_type = 'walking'
        if raw_tt[frame_id] == 1:
            current_frame_type = 'turning'

        # add white text on a red rectangle
        text = current_frame_type
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size
        margin = 5
        lower_left_corner = (100, 200 - text_height - margin)
        upper_right_corner = (100 + text_width + margin, 200 + margin)
        cv2.rectangle(frame, lower_left_corner, upper_right_corner, (0, 0, 255), cv2.FILLED)
        cv2.putText(
            frame,
            text,
            (100, 200),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

        out.write(frame)
    out.release()
    del out


def calculate_iou(
    bbox1: t.Tuple[int, int, int, int],
    bbox2: t.Tuple[int, int, int, int],
) -> float:
    """
    Calculate IOU of two bbox; each bbox in a format of (left, top, width, height)
    """
    # Unpack the bounding boxes
    left1, top1, width1, height1 = bbox1
    left2, top2, width2, height2 = bbox2

    # Calculate the bottom-right corners
    right1, bottom1 = left1 + width1, top1 + height1
    right2, bottom2 = left2 + width2, top2 + height2

    # Calculate intersection coordinates
    inter_left = max(left1, left2)
    inter_top = max(top1, top2)
    inter_right = min(right1, right2)
    inter_bottom = min(bottom1, bottom2)

    # Calculate intersection area
    inter_width = inter_right - inter_left
    inter_height = inter_bottom - inter_top
    if inter_width > 0 and inter_height > 0:  # Check if there is an intersection
        intersection_area = inter_width * inter_height
    else:
        intersection_area = 0

    # Calculate union area
    union_area = width1 * height1 + width2 * height2 - intersection_area

    # Calculate Intersection over Union (IoU)
    iou = intersection_area / union_area

    return iou


def get_target_boxes_and_keypoints_from_detection_2d(
    detectron_2d,
    targeted_person_bboxes,
):

    bb = detectron_2d['boxes']
    kp = detectron_2d['keypoints']
    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(None)  # 4 bounding box coordinates
            results_kp.append(None)  # 17 COCO keypoints
        elif len(targeted_person_bboxes[i]) == 0:
            results_bb.append(None)  # 4 bounding box coordinates
            results_kp.append(None)  # 17 COCO keypoints
        else:
            max_iou = 0
            potential_box_idx = None
            target_box = targeted_person_bboxes[i]
            for available_box_idx, available_box in enumerate(bb[i][1]):
                _left, _top, _right, _bottom, _ = available_box
                _available_bbox = (_left, _top, _right - _left, _bottom - _top)
                _iou = calculate_iou(target_box, _available_bbox)
                if _iou > max_iou:
                    max_iou = _iou
                    potential_box_idx = available_box_idx
            if max_iou < 0.5 or potential_box_idx is None:
                results_bb.append(None)  # 4 bounding box coordinates
                results_kp.append(None)  # 17 COCO keypoints
            else:
                best_match = potential_box_idx
                best_bb = bb[i][1][best_match, :4]
                best_kp = kp[i][1][best_match].T.copy()[:, [0, 1, 3]]
                results_bb.append(best_bb)
                results_kp.append(best_kp)
    return results_bb, results_kp


def render_detectron_2d_with_target_box(
    video_path: str,
    detectron_2d_path: str,
    targeted_person_bboxes_path: str,
    tt_pickle_path: str,
    output_video_path: str,
    draw_background: bool = True,
) -> None:
    with open(targeted_person_bboxes_path, 'rb') as handle:
        targeted_person_bboxes = pickle.load(handle)
    detectron_2d = np.load(detectron_2d_path, encoding='latin1', allow_pickle=True)

    _, keypoints = get_target_boxes_and_keypoints_from_detection_2d(
        detectron_2d=detectron_2d,
        targeted_person_bboxes=targeted_person_bboxes,
    )

    with open(tt_pickle_path, 'rb') as handle:
        raw_tt = pickle.load(handle)

    frames = []
    for frame in get_frames(video_path):
        if draw_background:
            frames.append(frame)
        else:
            frames.append(np.zeros_like(frame))

    image_height, image_width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (image_width, image_height))

    n = len(keypoints)
    for frame_id in range(n):
        frame = frames[frame_id]
        color = (0, 0, 255)  # red
        if not draw_background:
            color = (255, 255, 255)

        if len(targeted_person_bboxes[frame_id]) == 4:
            left, top, width, height = targeted_person_bboxes[frame_id]
            frame = cv2.rectangle(
                frame, [left, top], [left + width, top + height], (0, 255, 255), 10,  # yellow
            )

        if keypoints[frame_id] is not None and keypoints[frame_id][15][2] >= 0.2 and keypoints[frame_id][16][2] >= 0.2:  # noqa

            # for point in keypoints[frame_id]:
            #     if point[2] >= 0.2:
            #         cv2.circle(frame, (int(point[0]), int(point[1])), 10, color, -1)

            for (from_idx, to_idx) in gen_pairs([10, 8, 6, 5, 7, 9]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx][:2].astype(int)),
                    tuple(keypoints[frame_id][to_idx][:2].astype(int)),
                    color,
                    10,
                )

            for (from_idx, to_idx) in gen_pairs([6, 12, 14, 16]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx][:2].astype(int)),
                    tuple(keypoints[frame_id][to_idx][:2].astype(int)),
                    (255, 0, 255),  # purple; right
                    10,
                )

            for (from_idx, to_idx) in gen_pairs([6, 12]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx][:2].astype(int)),
                    tuple(keypoints[frame_id][to_idx][:2].astype(int)),
                    color,
                    10,
                )

            for (from_idx, to_idx) in gen_pairs([5, 11]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx][:2].astype(int)),
                    tuple(keypoints[frame_id][to_idx][:2].astype(int)),
                    color,
                    10,
                )

            for (from_idx, to_idx) in gen_pairs([11, 13, 15]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx][:2].astype(int)),
                    tuple(keypoints[frame_id][to_idx][:2].astype(int)),
                    (255, 255, 0),  # light blue; left
                    10,
                )

            for (from_idx, to_idx) in gen_pairs([12, 11]):
                cv2.line(
                    frame,
                    tuple(keypoints[frame_id][from_idx][:2].astype(int)),
                    tuple(keypoints[frame_id][to_idx][:2].astype(int)),
                    color,
                    10,
                )

        current_frame_type = 'walking'
        if raw_tt[frame_id] == 1:
            current_frame_type = 'turning'

        # add white text on a red rectangle
        text = current_frame_type
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size
        margin = 5
        lower_left_corner = (100, 200 - text_height - margin)
        upper_right_corner = (100 + text_width + margin, 200 + margin)
        cv2.rectangle(frame, lower_left_corner, upper_right_corner, (0, 0, 255), cv2.FILLED)
        cv2.putText(
            frame,
            text,
            (100, 200),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
        out.write(frame)
    out.release()
    del out
