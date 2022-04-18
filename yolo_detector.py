import cv2
import numpy as np


class DetectorYOLO():
    def __init__(self, weight_path, cfg_path, label_path, confidence_thresh, use_gpu):
        self.yolo_weight_path = weight_path[0]
        self.yolo_cfg_path = cfg_path[0]
        self.yolo_label_path = label_path[0]
        self.yolo_confidence_tresh = confidence_thresh
        self.gpu_flag = use_gpu
        self.side_face_detector = self.load_model()
        self.output_layers = self.get_output_layers()
        self.labels_list = self.read_label_file()

    def read_label_file(self):
        label_list = open(self.yolo_label_path).read().strip().split("\n")
        return label_list

    def load_model(self):
        # print("DEBUG_NAME: ", self.yolo_cfg_path, self.yolo_weight_path)
        net = cv2.dnn.readNetFromDarknet(self.yolo_cfg_path, self.yolo_weight_path)
        if self.gpu_flag:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #else:
        #    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        return net

    def get_output_layers(self):
        ln = self.side_face_detector.getLayerNames()
        # print("LN ", ln)
        ln = [ln[i[0] - 1] for i in self.side_face_detector.getUnconnectedOutLayers()]
        return ln

    def detect(self, frame):
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.side_face_detector.setInput(blob)
        layerOutputs = self.side_face_detector.forward(self.output_layers)
        return layerOutputs

    def filter_nms_yolo_detections(self, frame, yolo_outputs, draw_in_frame):
        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        (H, W) = frame.shape[:2]

        # loop over each of the layer outputs
        for output in yolo_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.yolo_confidence_tresh:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.yolo_confidence_tresh, 0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                if draw_in_frame:
                    # draw a bounding box rectangle and label on the image
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "{}: {:.4f}".format(self.labels_list[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                return {'classid': classIDs[i], 'confidence': confidences[i], 'bbox': (x, y, w, h)}

        else:
            return None