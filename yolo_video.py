# USAGE
# python yolo_video.py --input videos/car_back_video_2.mp4 --output output/output.avi --yolo yolo-coco --way l

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def make_coordinates(image, line_parameters, rcnt, lcnt):
    slope, intercept = line_parameters
    #기울기   절편
    try:
        y1 = image.shape[0] #height
        y2 = int(y1*(3/5)) #height의 3/5까지 올라갈 때까지 탐색...
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2]), rcnt, lcnt
    except OverflowError:
        return np.array([0, 0, 0, 0]), rcnt, lcnt


def average_slope_intercept(image, lines, rcnt, lcnt, maximum):
    left_fit = []  # 왼쪽 line 의 평균
    right_fit = []  # 오른쪽 line의 평균좌표
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if (-1) * maximum < slope < -0.2:  # line의 기울기가 음수인 경우
                left_fit.append((slope, intercept))
                lcnt = 0
            if maximum > slope > 0.2:  # line의 기울기가 양수인 경우
                right_fit.append((slope, intercept))
                rcnt = 0
            # if slope >= 2.0 or slope <= -2.0 :
            # print("can't detect!")
    except TypeError:
        left_fit_average = []
        right_fit_average = []

    # 위 모든 값들을 평균하여서 기울기와 y절편 계산하기
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    if left_fit_average != []:
        left_line, rcnt, lcnt = make_coordinates(image, left_fit_average, rcnt, lcnt)
    else:
        left_line = np.array([0, 0, 0, 0])
    if right_fit_average != []:
        right_line, rcnt, lcnt = make_coordinates(image, right_fit_average, rcnt, lcnt)
    else:
        right_line = np.array([0, 0, 0, 0])
    if right_fit == []:
        rcnt += 1
    else:
        rcnt = 0
    if left_fit == []:
        lcnt += 1
    else:
        lcnt = 0
    return np.array([left_line, right_line]), rcnt, lcnt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Second. Gaussian Blur (to Reduce Noise)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) #5x5 kernel size로 convolution 한다.
    #Apply Canny Method
    canny = cv2.Canny(blur, 10, 150) #급격한 변화만을 남겨 준다 (흰색으로 표시됨)
                                 #gradient 를 극단적으로 취해 준다
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None: # ==not empty
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) #2d array 를 1d array 로 변환
            #기울기
            s = (y2-y1)/(x2-x1)
            if 0.2 < abs(s) < 2.0 : cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]  # very simillar to 700
    width = image.shape[1]
    mwidth = width / 2
    polygons = np.array([
        [(int(width * 0 / 9), height), (int(width * 8 / 9), height), (int(mwidth), int(height * 5 / 9))]
    ])

    mask = np.zeros_like(image)  # 이미지와 같은 모양(pixel)으로 0배열을 만든다.
    cv2.fillPoly(mask, polygons, 255)  # 윤곽 만들기
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def ishide(image):
    avg = np.array([0, 0, 0])
    s = image.shape
    array_convert = np.array(image)
    width = s[1]
    height = s[0]
    for j in range(int(width*1/5), int(width*4/5)):
        for i in range(int(height*2/3), int(height)):
            bgr = array_convert[int(i), int(j)]
            avg[0] += bgr[2]
            avg[1] += bgr[1]
            avg[2] += bgr[0]
    for i in range(3):
        avg[i] = avg[i] / (width * (height / 3))
    return avg

def amplify(image):
    s = image.shape
    #img = image.shape
    width = s[1]
    height = s[0]
    im = np.array(cv2.imread('pic.jpg'))

    for j in range(int(width*4/5)):
        for i in range(int(height/2), int(height)):
            bgr = im[int(i), int(j)]
            if bgr[2] > 55 and bgr[2] < 120: im[int(i)][int(j)] = [255, 255, 255]
            #except IndexError:
    return im

########함수 정의 끝##############
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", required=True,
    help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument('-w', "--way", required=True,
    help="select a way to change car line")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")

args = vars(ap.parse_args())

changeway = args["way"]
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] Loading...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
rcnt = 0
lcnt = 0
color_avg_i = 0
maximum = 2.0
count = 0

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames detected.".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
carpos = []
carbeforepos = []
index = 0
maxx = 0
maxy = 0
maxi = 0
before_maxx = 0
before_maxy = 0
while True:
    carpos = []
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    flag = True
    cv2.imwrite('pic.jpg', frame)
    frame_o = frame
    hh = frame_o.shape[0] #height
    ww = frame_o.shape[1] #width
    maxi = 0
    mini = hh

    # if color_avg_i == 0: color_avg_i = sum(ishide(frame))
    # if abs(sum(ishide(frame)) - color_avg_i) >= 90:
    #    frame = amplify(frame)
    #    print('wait...amplifying')

    # Firstly, convert image to grayscale
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

    # Hough Transform (Region of Interest) and draw lines
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]),
                            minLineLength=40, maxLineGap=5)
    averaged_lines, rcnt, lcnt = average_slope_intercept(frame, lines, rcnt, lcnt, maximum)
    for item in averaged_lines:
        if max(item) > 10000: flag = False

    if flag == True:
        m1 = ((hh - averaged_lines[0][3])-(hh - averaged_lines[0][1]))/(averaged_lines[0][2]-averaged_lines[0][0])
        m2 = ((hh - averaged_lines[1][3])-(hh - averaged_lines[1][1]))/(averaged_lines[1][2]-averaged_lines[1][0])
        #절편을 계산
        b1 = -m1*averaged_lines[0][0] + (hh - averaged_lines[0][1])
        b2 = -m2*averaged_lines[1][0] + (hh - averaged_lines[1][1])
        line_image = display_lines(frame, averaged_lines)

        # averaged_lines를 통해서 line을 조금 더 부드럽게 만들기
        combo_image = cv2.addWeighted(frame_o, 0.8, line_image, 1, 1)
        # 가중치 부여해 이미지 합치기
        color = [255, 0, 0]
        if rcnt >= 5:
            if rcnt % 5 == 0:
                print('오른쪽 점선')
                # cv2.putText(combo_image, '오른쪽 점선', (100, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if lcnt >= 5:
            if lcnt % 5 == 0:
                print('왼쪽 점선')
                # cv2.putText(combo_image, '왼쪽 점선', (60, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            newy = hh - (y + h * 0.5)
            newx = x + 0.5 * w
            if changeway == 'l':
                if newx < (7/9) * ww and (newy - m1*newx - b1) < 0 and (newy - m2*newx - b2) > 0 and (LABELS[classIDs[i]] == 'car' or LABELS[classIDs[i]] == 'truck'):
                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(combo_image, (x, y), (x + w, y + h), color, 1)
                    text1 = "{}".format(LABELS[classIDs[i]]) #	confidences[i]
                    text2 = "pos = " + str(newx) + ", " + str(newy)
                    texttotal = str(round((hh-newy) - m1*x-b1, 2)) + ', ' + str(round((hh-newy) - m2*x-b2, 2))
                    cv2.putText(combo_image, text2, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    carpos.append([newx, newy])
            elif changeway == 'r':
                if newx > (2/9) * ww and (newy - m1*newx - b1) > 0 and (newy - m2*newx - b2) < 0 and (LABELS[classIDs[i]] == 'car' or LABELS[classIDs[i]] == 'truck'):
                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(combo_image, (x, y), (x + w, y + h), color, 1)
                    text1 = "{}".format(LABELS[classIDs[i]]) #	confidences[i]
                    text2 = "pos = " + str(x) + ", " + str(y)
                    texttotal = str(round((hh-newy) - m1*x-b1, 2)) + ', ' + str(round((hh-newy) - m2*x-b2, 2))
                    cv2.putText(combo_image, text2, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    carpos.append([newx, newy])

        #carpos 에서 특정 값만 취하기
        for i in range(len(carpos)):
            if mini > carpos[i][1]:
                mini = carpos[i][1]
                index = i
        if carpos != []:
            maxx = carpos[index][0]
            maxy = carpos[index][1]
        #print('all_carpos:', carpos)
        print('car :', maxx, maxy)

        #상대 좌표 변화값 (상대속도) 구하기
        if before_maxy != 0 and before_maxx != 0:
            dx = before_maxx - maxx
            dy = before_maxy - maxy
            print('∆x =', dx, ' ∆y =', dy)
            dpos = np.sqrt((dx)**2 + (dy)**2)
            print('∆pos = ', dpos)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    #writer.write(frame)
    count += 1
    if count == 700: break

    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) == ord('q'): break

    before_maxx = maxx
    before_maxy = maxy

# release the file pointers
print("[INFO] Closing...")
#writer.release()
vs.release()