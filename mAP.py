import numpy as np

sorted_ind = np.argsort(-confidence)
BB = BB[sorted_ind, :] # predict bounding box coordinates
image_ids = [image_ids[x] for x in sorted_ind] # get image id of prediction
                                               #bounding box
# retrieve prediction bbox
nd = len(image_ids)
tp = np.zeros(nd)
fp = np.zeros(nd)
for d in range(nd):
    R = class_recs[image_ids[d]]
    bb = BB[d, :].astype(float)
    ovmax = -np.inf 
    BBGT = R['bbox'].astype(float)
    
    if BBGT.size > 0:
        # calculate IOU
        # first calculate intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        
        # then calculate union
        uni = ((bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + 
                (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
    
    # get max IOU
    if ovmax > ovthresh: # confidence threshold
        if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
    else:
        fp[d] = 1.

# calculate precision and recall
fp = np.cumsum(fp)
tp = np.cumsum(tp)
rec = tp / float(npos)
prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

# get different rank values -> precison and recall
def voc_ap(rec, prec, use_07_metric=False):
    """
    If use_07_metric is true, use the VOC 07 11-point method.
    """
    if use_07_metric:
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
    """
    new method: calculate all points
    """
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision 
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points 
        # where X axis(recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

