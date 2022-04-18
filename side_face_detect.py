import cv2
from yolo_detector import DetectorYOLO
import glob

detector_side_face = DetectorYOLO(["models/yolov4-large-two-class_final.weights"], ["models/yolov4-large-two-class.cfg"],
                            ["models/labels.txt"], 0.85, True)

detector_frontal_face = DetectorYOLO(["models/yolov4-tiny-two-class_best.weights"], ["models/yolov4-tiny-two-class.cfg"],
                            ["models/labels.txt"], 0.70, True)

annot_file_csv = open("annotation.csv", "w")
annot_file_csv.write("filename,x,y,top,left,class\n")

ids = [1, 2, 3]
for id in ids:
    label_file = open("label_files/annotation_0" + str(id) + ".txt")
    file_names = label_file.readlines()
    aa = 0
    for name in file_names:
        fold_name = name.split(";")[0].split(".")[0]
        print("[", aa, len(file_names), "] ", "Analisando pasta: ", fold_name)
        if name.split(";")[1].strip("\n") == "lateraldireita":
            path = "./frames_labeled/lateral_direita/"
            for frame_name in glob.glob((path + fold_name + "/*.jpg")):
                frame = cv2.imread(frame_name)
                raw_det = detector_side_face.detect(frame)
                prediction = detector_side_face.filter_nms_yolo_detections(frame, raw_det, False)
                if prediction is not None:
                    if prediction['classid'] == 0:
                        # print("Not None dir")
                        frontal_raw_det = detector_frontal_face.detect(frame)
                        frontal_pred = detector_side_face.filter_nms_yolo_detections(frame, frontal_raw_det, False)
                        if frontal_pred is not None:
                            if frontal_pred['classid'] != 1:
                                # cv2.imwrite("./frames_labeled/valid_faces/lateral_direita/" + frame_name.split("/")[-1], frame)
                                bbox = prediction['bbox']
                                name_write = frame_name.split("/")[-1] + "," + str(bbox[0]) + "," + \
                                             str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]) + "," + "lateral_direita\n"
                                annot_file_csv.write(name_write)
        elif name.split(";")[1].strip("\n") == "lateralesquerda":
            path = "./frames_labeled/lateral_esquerda/"
            for frame_name in glob.glob((path + fold_name + "/*.jpg")):
                frame = cv2.imread(frame_name)
                raw_det = detector_side_face.detect(frame)
                prediction = detector_side_face.filter_nms_yolo_detections(frame, raw_det, False)
                if prediction is not None:
                    if prediction['classid'] == 0:
                        # print("Not None esq")
                        frontal_raw_det = detector_frontal_face.detect(frame)
                        frontal_pred = detector_side_face.filter_nms_yolo_detections(frame, frontal_raw_det, False)
                        if frontal_pred is not None:
                            if frontal_pred['classid'] != 1:
                                # cv2.imwrite("./frames_labeled/valid_faces/lateral_esquerda/" + frame_name.split("/")[-1], frame)
                                bbox = prediction['bbox']
                                name_write = frame_name.split("/")[-1] + "," + str(bbox[0]) + "," + \
                                             str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]) + "," + "lateral_esquerda\n"
                                annot_file_csv.write(name_write)
        aa += 1

annot_file_csv.close()
