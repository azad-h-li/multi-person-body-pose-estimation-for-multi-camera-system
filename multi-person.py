import cv2
import mediapipe
import math

# class names from coco.names will be copied to classNames
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# introduction mobilenet and graph files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# creating model
net1 = cv2.dnn_DetectionModel(weightsPath, configPath)
# these configurations below are from a documentation
net1.setInputSize(320, 320)
net1.setInputScale(1.0 / 127.5)
net1.setInputMean((127.5, 127.5, 127.5))
net1.setInputSwapRB(True)


def segment(img):
    objects = ['person']
    # getting the prediction form the image
    classIds, confs, bbox = net1.detect(img, 0.6, 0.2)
    if len(classIds) != 0:
        # taking one id from the classids (clasids are lattened to be used)
        for classId, box in zip(classIds.flatten(), bbox):
            # classId starts from one, but array starts from 0, that’s why 1 is substracted from classId
            className = classNames[classId - 1]
            # if the detected object is desired object
            landm=[]
            if className in objects:
                # draw the rectangle box on the image
                x, y, w, h = box
                #cv2.rectangle(img, (x, y), (x + w, y + h), (200, 50, 100), 2)
                roi = img[y:y+h, x:x+w]
                img, landm = detector.findPose(img, roi, x, y)
            if (len(landm)!= 0):
                text = pose_est(landm)
                cv2.putText(img, text, (int(x+w/2), y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
    return img


class poseDetector():
    def __init__(self, mode=False, modelComplex=0, smooth_lm=True,
                 en_seg=False, smooth_seg=True, detection_con=0.7, track_con=0.5):
        self.mode = mode
        self.modelComplex = modelComplex
        self.smooth_lm = smooth_lm
        self.en_seg = en_seg
        self.smooth_seg = smooth_seg
        self.detection_con = detection_con
        self.track_con = track_con
        self.mediapipe_draw = mediapipe.solutions.drawing_utils
        self.mediapipe_pose = mediapipe.solutions.pose
        self.pose = self.mediapipe_pose.Pose(self.mode, self.modelComplex, self.smooth_lm, self.en_seg,
                                             self.smooth_seg, self.detection_con, self.track_con)

    def findPose(self, img, roi, x1, y1):
        roiRGB = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        h_roi, w_roi, _ = roi.shape
        h_im, w_im, _ = img.shape
        self.results = self.pose.process(roiRGB)
        landmarks_m = []
        if self.results.pose_landmarks:
            id = 0
            while id < 32:
                lmx = self.results.pose_landmarks.landmark[id].x * w_roi + x1
                self.results.pose_landmarks.landmark[id].x = lmx / w_im
                lmy = self.results.pose_landmarks.landmark[id].y * h_roi + y1
                self.results.pose_landmarks.landmark[id].y = lmy / h_im
                landmarks_m.append([id, int(lmx), int(lmy)])
                id = id + 1
            self.mediapipe_draw.draw_landmarks(img, self.results.pose_landmarks, self.mediapipe_pose.POSE_CONNECTIONS)
        return img, landmarks_m


def find_angle(landm, l1, l2, l3):
    x1, y1 = landm[l1][1:]
    x2, y2 = landm[l2][1:]
    x3, y3 = landm[l3][1:]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return angle


def in_range(landm):
    x, y = landm[1:]
    if (0 <= x <= 640 and 0 <= y <= 540):
        return 1
    return 0


def pose_est(landm):
    text = "unknown"
    if in_range(landm[12]) == 1 and in_range(landm[24]) == 1 and in_range(landm[26]) == 1:
        ang1 = abs(find_angle(landm, 12, 24, 26))
        if in_range(landm[11]) == 1 and in_range(landm[23]) == 1 and in_range(landm[25]) == 1:
            ang2 = abs(find_angle(landm, 11, 23, 25))
            if ang1 > 180:
                ang1 = abs(360 - ang1)
                print(ang1)
            if ang2 > 180:
                ang2 = abs(360 - ang2)
                print(ang2)
            if 120 > ang1 and 120 > ang2:
                text = "sitting"
            elif 150 < ang1 and 150 < ang2:
                text = "standing"
        else:
            if 120 > ang1:
                text = "sitting"
            elif 150 < ang1:
                text = "standing"
    elif (in_range(landm[11]) and in_range(landm[23]) and in_range(landm[25])):
        ang2 = abs(find_angle(landm, 11, 23, 25))
        if ang2 > 180:
            ang2 = abs(360 - ang2)
            print(ang2)
        if 120 > ang2:
            text = "sitting"
        elif 150 < ang2:
            text = "standing"
    return text


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    detector = poseDetector()
    while cam.isOpened():
        success, img = cam.read()
        img = cv2.flip(img, 1, 0)
        img = segment(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
cam.release
cv2.destroyAllWindows()
