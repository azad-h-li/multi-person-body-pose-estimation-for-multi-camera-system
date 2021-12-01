import cv2
import mediapipe
import math
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


    def find_config(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            self.mediapipe_draw.draw_landmarks(img, self.results.pose_landmarks, self.mediapipe_pose.POSE_CONNECTIONS)
        return img

    def find_keypoint(self, img):
        self.landmarks_m = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks_m.append([id, cx, cy])
        return self.landmarks_m

def find_angle(landm, l1, l2, l3):
    x1, y1 = landm[l1][1:]
    x2, y2 = landm[l2][1:]
    x3, y3 = landm[l3][1:]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return angle

def in_range(landm):
    x, y = landm[1:]
    if ( 0 <= x <= 640 and 0 <= y <= 480):
        return 1
    return 0

def pose_est(landm):
    text = "unknown"
    if in_range(landm[12]) == 1 and in_range(landm[24]) == 1 and in_range(landm[26]) == 1:
        ang1 = abs(find_angle(landm, 12, 24, 26))
        if in_range(landm[11]) == 1 and in_range(landm[23]) == 1 and in_range(landm[25]) == 1:
            ang2 = abs(find_angle(landm, 11, 23, 25))
            if ang1 > 180:
                ang1 = abs(360-ang1)
                print(ang1)
            if ang2 > 180:
                ang2 = abs(360-ang2)
                print(ang2)
            if 120 > ang1 and 120 > ang2:
                text ="sitting"
            elif 150 < ang1 and 150 < ang2:
                text ="standing"
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
        if 131 > ang2:
            text = "sitting"
        elif 150 < ang2:
            text = "standing"
    return text
    cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    detector = poseDetector()
    while cam.isOpened():
        success, img = cam.read()
        img = cv2.flip(img, 1, 0)
        img = detector.find_config(img)
        landm = detector.find_keypoint(img)
        if (landm):
            text = pose_est(landm)
            cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cam.release
    cv2.destroyAllWindows()
