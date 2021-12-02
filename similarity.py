import cv2
from skimage.metrics import structural_similarity
from skimage.transform import resize

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
        roi = []
        for classId, box in zip(classIds.flatten(), bbox):
            # classId starts from one, but array starts from 0, that’s why 1 is substracted from classId
            className = classNames[classId - 1]
            # if the detected object is desired object

            if className in objects:
                roi.append(box)
    return img, roi, len(roi)

def orb_sim(img1, img2):
    # SIFT is no longer available in cv2 so using ORB
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # perform matches.
    matches = bf.match(desc_a, desc_b)
    # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

roi1 = []
img1 = cv2.imread("F:\depositphotos_180482706-stock-photo-3-people-work-in-the.jpg")
img1, roi1, length1 = segment(img1)
roi2 = []
img2 = cv2.imread("F:\depositphotos_179714592-stock-photo-3-people-work-in-the.jpg")
img2, roi2, length2 = segment(img2)
match = []
for i in range(length1):
    max_similarity = 0
    similar = -1
    for j in range(length2):
        im_i = img1[roi1[i][1]:(roi1[i][1]+roi1[i][3]), roi1[i][0]:(roi1[i][0]+roi1[i][2])]
        im_j = img2[roi1[j][1]:(roi1[j][1]+roi1[j][3]), roi1[j][0]:(roi1[j][0]+roi1[j][2])]
        orb_similarity = orb_sim(im_i, im_j)
        print("Similarity between", i+1, "and", j+1, "using ORB is: ", orb_similarity)
        if (orb_similarity > max_similarity):
            max_similarity = orb_similarity
            similar = j
    if similar > -1:
        match.append([i, similar])
for pair in match:
    text = "person " + str(pair[0]+1)
    cv2.putText(img1, text, (roi1[pair[0]][0], roi1[pair[0]][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.putText(img2, text, (roi2[pair[1]][0], roi2[pair[1]][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2,
                cv2.LINE_AA)
print(match)
cv2.imshow("pic1", img1)
cv2.imshow("pic2", img2)
cv2.imwrite("per1.jpg", img1)
cv2.imwrite("per2.jpg", img2)
cv2.waitKey(7000)
cv2.destroyAllWindows()
