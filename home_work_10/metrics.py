def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    box = [bbox1, bbox2]
    num_left = 0 if min(bbox1[0], bbox2[0]) == bbox1[0] else 1
    if box[num_left][2] < box[1-num_left][0]:
        return 0
    num_low = 0 if min(bbox1[1], bbox2[1]) == bbox1[1] else 1
    if box[num_low][3] < box[1-num_low][1]:
        return 0
    intersection = (min(box[num_left][2], box[1-num_left][2]) - box[1-num_left][0]) * (min(box[num_low][3], box[1-num_low][3]) - box[1-num_low][1])
    return intersection / ((bbox1[2]- bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2]- bbox2[0]) * (bbox2[3] - bbox2[1]) - intersection)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj_dict = {val[0]: val[1:] for val in frame_obj}
        frame_hyp_dict = {val[0]: val[1:] for val in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for key, val in matches.items():
            if key in frame_obj_dict and val in frame_hyp_dict:
                score = iou_score(frame_hyp_dict[val], frame_obj_dict[key])
                if score > threshold:
                    dist_sum += score
                    match_count += 1
                    del frame_hyp_dict[val]
                    del frame_obj_dict[key]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        IOU_list = []
        for key, val in frame_hyp_dict.items():
            for k, v in frame_obj_dict.items():
                score = iou_score(val, v)
                if score > threshold:
                    IOU_list.append((score, key, k))
        IOU_list = sorted(IOU_list, key=lambda x: x[0], reverse=True)
        while len(IOU_list) > 0:
            top = IOU_list[0]
            matches.update({top[2]: top[1]})
            match_count += 1
            dist_sum += top[0]
            IOU_list = [t for t in IOU_list if (t[1] != top[1] and t[2] != top[2])]
        # Step 5: Update matches with current matched IDs

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        pass

    # Step 8: Calculate MOTP and MOTA
    MOTP = ...
    MOTA = ...

    return MOTP, MOTA
