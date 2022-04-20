import cv2
from yolo_detector import DetectorYOLO
import glob

detector_side_face = DetectorYOLO(["models/yolov4-large-two-class_final.weights"], ["models/yolov4-large-two-class.cfg"],
                            ["models/labels.txt"], 0.85, True)

annot_file_csv = open("annotation_correct.csv", "w")
annot_file_csv.write("filename,x,y,top,left,class\n")

path_names = ["./frames_labeled/Lateral_face_filtered/lateral_direita/",
              "./frames_labeled/Lateral_face_filtered/lateral_esquerda/"]

error_log_file = open("error_log.txt", "w")

for path in path_names:
    aa = 0
    for img_path in glob.glob(path + "*.jpg"):

        if path.split("/")[-2] == "lateral_direita":
            print("[", aa, len(glob.glob(path + "*.jpg")), "] ", "Analisando imagem: ", path.split("/")[-2], "/",
                  img_path.split("/")[-1])
            frame = cv2.imread(img_path)
            raw_det = detector_side_face.detect(frame)
            prediction = detector_side_face.filter_nms_yolo_detections(frame, raw_det, False)
            if prediction is not None:
                if prediction['classid'] == 0:
                    bbox = prediction['bbox']
                    name_write = img_path.split("/")[-1] + "," + str(bbox[0]) + "," + \
                                 str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]) + "," + "lateral_direita\n"
                    annot_file_csv.write(name_write)
            else:
                error_log_file.write(img_path + "\n")

        elif path.split("/")[-2] == "lateral_esquerda":
            print("[", aa, len(glob.glob(path + "*.jpg")), "] ", "Analisando imagem: ", path.split("/")[-2], "/",
                  img_path.split("/")[-1])
            frame = cv2.imread(img_path)
            raw_det = detector_side_face.detect(frame)
            prediction = detector_side_face.filter_nms_yolo_detections(frame, raw_det, False)
            if prediction is not None:
                if prediction['classid'] == 0:
                    bbox = prediction['bbox']
                    name_write = img_path.split("/")[-1] + "," + str(bbox[0]) + "," + \
                                 str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]) + "," + "lateral_esquerda\n"
                    annot_file_csv.write(name_write)

            else:
                error_log_file.write(img_path + "\n")
        aa += 1

annot_file_csv.close()
