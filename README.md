# infrastructure

## mAP calculation
#### mAP calculation for VOC 
* use IOU = 0.5(default)
* use 11 equally sapced recall levels[0, 0.1, 0.2, ..., 1.0]
* use confidence descending order

#### mAP calculation for COCO

## tracker logic
tracks: a list of tracker(custom class)

dets: a list of detection bounding boxs
* check if tracks is empty
  * tracks empty -> init tracks
  * tracks not empyt -> update tracks
* match tracks and dets
  * if a tracker match failure, it will preserve **unless it fails more than skippedFrames**
  * if a detection match failure, it will new a tracker use this detection
  * if a tracker match success, it will update use correspond detection and **set skippedFrames to 0**

## dataset preprocess
pre-process caffe order
* convert to single(float32 type)
* resize image size
* transpose dimensions to KxHxW
* reorder channels (color to BGR)
* scale raw input (from [0, 1] to [0, 255])
* **subtract mean**
* **scale feature**

## post-process
detection post-process
* nms series
  * nms
  * soft-nms(code provided)
  * softer-nms(code provided)
