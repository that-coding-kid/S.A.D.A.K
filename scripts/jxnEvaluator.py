#pip install supervision==0.20.0
#pip install ultralytics==8.1.30
# %cd {HOME}
# !python -m wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pz68D1Gsx80MoPg-_q-IbEdESEmyVLm-" -O vehicle-counting.mp4 && rm -rf /tmp/cookies.txt
#!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# %cd {HOME}
# %cd {'C://Users//Dell//Desktop//Datathon//sadakDeployed//sadakInternal//ByteTrack'}


#git clone https://github.com/ifzhang/ByteTrack.git
#sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt (enter into ByteTrackFolder First)
#pip3 install -r requirements.txt
#python setup.py develop
#pip install cython_bbox
#pip install onemetric
# workaround related to https://github.com/roboflow/notebooks/issues/112 and https://github.com/roboflow/notebooks/issues/106
#pip install loguru lap thop
#%cd {'C://Users//Dell//Desktop//Datathon//sadakDeployed//sadakInternal'}

import numpy as np
import supervision as sv
import ultralytics
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import time
from supervision.utils.video import VideoSink
from supervision.utils.video import get_video_frames_generator
from supervision.detection.core import Detections
from supervision.utils.video import VideoInfo
from supervision.draw.color import ColorPalette
from supervision.annotators.utils import ColorLookup
from scipy.signal import resample
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from IPython import display
from ultralytics import YOLO
from typing import Any,Optional
import streamlit as st

ultralytics.checks()

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

print("supervision.__version__:", sv.__version__)

#later yolo8x if possible
model = YOLO(f"../yolov8n.pt")

uniqueColors = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]
uniqueColorsFinal = ColorPalette.from_hex(uniqueColors)

# frameThreshold = 24
# regressionPower = 4
# stdDevThreshold = 0

"""FrameThreshold: All the trackerIDs that show up in less frames than frameThreshold would be dropped. <br>
RegressionPower: Decides the degree of polynomial regression. <br>
StdDevThreshold: Drops the tracking IDs whose position change slower than this threshold"""

COLORS = sv.ColorPalette.DEFAULT
THICKNESS = 2
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]

SOURCE_VIDEO_PATH = f"videos/junctionEvalDataset/ChickPeaClips/clip0.mp4"
TARGET_VIDEO_PATH = f"scripts/deleteLater.mp4"

byte_tracker = sv.ByteTrack()

class idInstances:
    def __init__(self):
        self.classId = {}
        self.frames = []
        self.bboxes = []
        self.stdDev = 0
        self.motionId = 0

class polynomial_reg:

    def __init__(self,inp_data,inp_labels,power,test_data,test_labels,max_att):  
        self.x_vec = inp_data
        self.y_vec = inp_labels
        self.p = power
        self.t_x_vec = test_data
        self.t_y_vec = test_labels
        self.att_name = max_att

    def z_matrix(self):
        self.z = []
        for i in range(0,len(self.x_vec)):
            row = []
            for j in range(self.p+1):
                row.append(self.x_vec[i]**j)
            self.z.append(row)
        self.z = np.array(self.z)
    
    def weight_vector(self):
        val1 = np.linalg.inv(np.dot(self.z.T,self.z))
        val2 = np.dot(val1,self.z.T)
        self.w_vec = np.dot(val2,self.y_vec)
    
    def predictions(self):
        self.predictions_train = []
        for i in range(0,len(self.x_vec)):
            prediction = 0
            for j in range(0,self.p+1):
                prediction+=(self.w_vec[j]*((self.x_vec[i])**j))
            self.predictions_train.append(prediction)
        self.predictions_test = []
        for i in range(0,len(self.t_x_vec)):
            prediction = 0
            for j in range(0,self.p+1):
                prediction+=(self.w_vec[j]*((self.t_x_vec[i])**j))
            self.predictions_test.append(prediction)
        self.predictions_test= np.array(self.predictions_test)
        self.predictions_test = np.array(self.predictions_test)
    
    def rmse(self):

        rmse_train = self.predictions_train - self.y_vec
        rmse_train = rmse_train**2
        sum_val = sum(rmse_train)
        val = sum_val/len(rmse_train)
        val = val**0.5
        self.rv_tr = val
        self.rv_tr_p = (val/self.y_vec.mean())*100
        rmse_test = self.predictions_test - self.t_y_vec
        rmse_test = rmse_test**2
        sum_val = sum(rmse_test)
        val = sum_val/len(rmse_test)
        val = val**0.5
        self.rv_tt = val
        self.rv_tt_p = (val/self.t_y_vec.mean())*100
            
    def plotter_train(self):
        x_axis = np.arange(min(self.x_vec),max(self.x_vec),0.01)
        ones = np.ones(len(x_axis))
        y_axis= np.zeros(len(x_axis))
        for i in range(0,self.p+1):
            y_axis+=(self.w_vec[i]*(x_axis**i))
        plt.plot(x_axis,y_axis,linewidth=2)

    def getValue(self,xval):
        yval = 0
        for i in range(0,self.p+1):
            yval+=(self.w_vec[i]*(xval**i))
        return yval

    def plotter_test(self):
        x_axis = np.arange(min(self.t_x_vec),max(self.t_x_vec),0.01)
        ones = np.ones(len(x_axis))
        y_axis= np.zeros(len(x_axis))
        for i in range(0,self.p+1):
            y_axis+=(self.w_vec[i]*(x_axis**i))
        plt.plot(x_axis,y_axis,linewidth=2)

def get_detections(SOURCE_VIDEO_PATH):
    overallDetections = []
    start_time = time.time()
    source_video_info = VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    progress_bar = st.empty()
    with VideoSink(target_path="", video_info=source_video_info) as sink:
        for index, frame in enumerate(
            get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
        ):  
            curr_time = time.time()
            elapsed_time = curr_time-start_time
            left_time = elapsed_time/(index+1)
            left_time = round(left_time*(source_video_info.total_frames - (index+1)),2)
            percentage = round((index+1)/source_video_info.total_frames,2)
            progress_bar.progress(percentage,"ETA: "+str(left_time) +" seconds")
            # print("Progress: "+str(index)+"/"+str(source_video_info.total_frames)+"; ETA: "+str(left_time))
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)
            overallDetections.append(detections)
    return overallDetections

def loadDetections(filename):
    with open(filename, 'rb') as file:
        loadedDetections = pickle.load(file)
    return loadedDetections

def saveDetections(detections,filename):
    with open(filename, 'wb') as file:
        pickle.dump(detections, file)
    return

def cleaningDataset(detections,frameThreshold):
    idDataset = {}
    currFrameNo = 0
    for frame in detections:
        for id in frame:
            if (id[4] in idDataset.keys()):
                idDataset[id[4]].frames.append(currFrameNo)
                idDataset[id[4]].bboxes.append(id[0])
                if (id[3] in idDataset[id[4]].classId.keys()):
                    idDataset[id[4]].classId[id[3]]+=id[2]
                else:
                    idDataset[id[4]].classId[id[3]] = id[2]
            else:
                idDataset[id[4]] = idInstances()
                idDataset[id[4]].classId[id[3]] = id[2]
                idDataset[id[4]].frames.append(currFrameNo)
                idDataset[id[4]].bboxes.append(id[0])
        currFrameNo+=1

    for id in idDataset:
        currMaxId = -1
        currMax = 0
        for key in idDataset[id].classId.keys():
            if (idDataset[id].classId[key] > currMax):
                currMax = idDataset[id].classId[key]
                currMaxId = key
        idDataset[id].classId = currMaxId

    finalDict = {}
    permittedIds = [1,2,3,5,6]
    for i in idDataset.keys():
        if (len(idDataset[i].frames) < frameThreshold):
            pass
        elif (idDataset[i].classId not in permittedIds):
            pass
        else:
            finalDict[i] = idDataset[i]

    return finalDict

def resize_to_screen(image, screen_width, screen_height):
    img_height, img_width = image.shape[:2]
    scale = min(screen_width / img_width, screen_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    return cv2.resize(image, (new_width, new_height))

def drawRectangles(SOURCE_VIDEO_PATH,SCREEN_W,SCREEN_H):

    def draw_rectangle(event, x, y, flags, param):
        global current_rectangle, drawing,rectangles
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_rectangle = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = img_resized.copy()
                cv2.rectangle(img_copy, current_rectangle[0], (x, y), (0, 255, 0), 2)
                for i in rectangles:
                    cv2.rectangle(img_copy,(i[0],i[1]),(i[2],i[3]),(0,255,0),2)
                cv2.imshow('image', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_rectangle.append((x, y))
            rectangles.append((current_rectangle[0][0], current_rectangle[0][1], x, y))
            img_copy = img_resized.copy()
            cv2.rectangle(img_copy, current_rectangle[0], (x, y), (0, 255, 0), 2)
            for i in rectangles:
                cv2.rectangle(img_copy,(i[0],i[1]),(i[2],i[3]),(0,255,0),2)
            cv2.imshow('image', img_copy)

    rectangles = []
    current_rectangle = []
    drawing = False

    video_path = SOURCE_VIDEO_PATH
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    ret, img = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    aspectRatio = img.shape[1]/img.shape[0]
    screen_width = SCREEN_W
    screen_height = SCREEN_W*aspectRatio

    img_resized = resize_to_screen(img, screen_width, screen_height)

    cv2.imshow('image', img_resized)
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print('Rectangles:', rectangles)
        elif key == 27:  
            break
        elif key == 8:  
            if rectangles:
                rectangles.pop()
                img_resized_copy = img_resized.copy()
                for rect in rectangles:
                    cv2.rectangle(img_resized_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                cv2.imshow('image', img_resized_copy)
        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break


    cv2.destroyAllWindows()

    widthCoeff = img.shape[1]/screen_width
    heightCoeff = img.shape[0]/screen_height
    rectangles_copy = np.array(rectangles)
    rectangles_copy = rectangles_copy.T
    rectangles_copy[0] = rectangles_copy[0]*widthCoeff
    rectangles_copy[2] = rectangles_copy[2]*widthCoeff
    rectangles_copy[1] = rectangles_copy[1]*heightCoeff
    rectangles_copy[3] = rectangles_copy[3]*heightCoeff

    return rectangles_copy

def obtainRectangles(SOURCE_VIDEO_PATH,finalDict,numClusters):
    startingPoints = []
    endingPoints = []
    # finalDict = {} #Remove later
    for i in finalDict.values():
        startingPt = ((i.bboxes[0][0]+i.bboxes[0][2])/2,(i.bboxes[0][1]+i.bboxes[0][3])/2)
        endingPt = ((i.bboxes[-1][0]+i.bboxes[-1][2])/2,(i.bboxes[-1][1]+i.bboxes[-1][3])/2)
        startingPoints.append(startingPt)
        endingPoints.append(endingPt)
    allpoints = startingPoints+endingPoints
    startingPoints = np.array(startingPoints)
    startingPoints = startingPoints.T
    plt.gca().invert_yaxis()
    plt.scatter(startingPoints[0],startingPoints[1])
    endingPoints = np.array(endingPoints)
    endingPoints = endingPoints.T
    plt.scatter(endingPoints[0],endingPoints[1])
    plt.legend(["starting","ending"])
    plt.show()
    # randomIndices = np.random.choice(range(0,len(startingPoints)),size=numClusters)

    # randomIndices = np.random.choice(range(0,len(startingPoints)),size=numClusters)
    # randomIndices = np.random.choice(range(0,len(startingPoints)),size=numClusters)

def fillMissingFrames(finalDict,REGRESSION_POWER):
    for i in finalDict.keys():
        print(finalDict[i])
    print(len(finalDict.keys()))
    for key in finalDict.keys():
        frames_axis = finalDict[key].frames
        x_axis_val = np.array(finalDict[key].frames)/len(finalDict[key].frames)
        y_axis_val = np.array(finalDict[key].bboxes)
        y_axis_val = y_axis_val.T
        # y_axis_valx1 = y_axis_val[0]
        # y_axis_valy1 = y_axis_val[1]
        # y_axis_valx2 = y_axis_val[2]
        # y_axis_valy2 = y_axis_val[3]

        instanceList = []
        try:
            for i in range(0,4):
                instance = polynomial_reg(x_axis_val,y_axis_val[i],REGRESSION_POWER,x_axis_val,y_axis_val[i],"max_att")
                instance.z_matrix()
                instance.weight_vector()
                instance.predictions()
                instance.rmse()
                instanceList.append(instance)
        except:
            continue
        finalLists = [[] for _ in range(4)]



        startFrame = frames_axis[0]
        endFrame = frames_axis[len(frames_axis)-1]+1


        # p1 = 0
        # print(startFrame)
        # print(endFrame)

        for j in range(0,4):
            p1=0
            for i in range(startFrame,endFrame):
                if (x_axis_val[p1] == i):
                    finalLists[j].append(y_axis_val[j][p1])
                    p1+=1
                    continue
                yval = instanceList[j].getValue(i/len(x_axis_val))
                finalLists[j].append(yval)
            
        # print(len(x_axis_val))

        finalLists = np.array(finalLists)
        finalLists = finalLists.T
        finalDict[key].frames = range(startFrame,endFrame)
        finalDict[key].bboxes = finalLists
        finalDict[key].confidence = np.array([0.5]*(endFrame-startFrame))
    return finalDict

def writeToVideo(finalDict,detections,SOURCE_VIDEO_PATH,TARGET_VIDEO_PATH,numFrames):

    for k in finalDict.keys():
        print("MotionID:", finalDict[k].motionId)

    source_video_info = VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)

    bboxes = [[] for _ in range(len(detections))]
    mask = [[] for _ in range(len(detections))]
    confidence = [[] for _ in range(len(detections))]
    classId = [[] for _ in range(len(detections))]
    trackerId = [[] for _ in range(len(detections))]
    classIdByName = [[] for _ in range(len(detections))]




    print("\nStarting second loop...\n")

    # Second loop: Populate the lists/dictionaries and print MotionID
    for tId in finalDict.keys():
        for i in range(len(finalDict[tId].frames)):
            frame_index = finalDict[tId].frames[i]

            # Populate the lists/dictionaries for each frame
            bboxes[frame_index].append(finalDict[tId].bboxes[i])
            classId[frame_index].append(finalDict[tId].classId)
            confidence[frame_index].append(0.5)
            trackerId[frame_index].append(finalDict[tId].motionId)
            classIdByName[frame_index].append(model.model.names[finalDict[tId].classId])

            # Print the MotionID for debugging
            print("Frame:", frame_index, "MotionID:", finalDict[tId].motionId)
    #bbox,mask,confidence,classId,trackerId,{'class_name':'person'}

    detections_arr = []
    #for a frame
    #xyxy = array 2 dimensional, dtype = float 32
    #mask = None (singular)
    #confidence = array 1 dimensional, dtype = float 32
    #classId = array 1 dimensional
    #trackerId = array 1 dimensional
    #data = {'class_name': array of classes, dtype = <U13}
    for i in range(0,len(bboxes)):
        newInstance = Detections(np.array(bboxes[i],dtype='float32'))
        # newInstanceMask = newInstance.xyxy.astype('int32')
        # currMask = bbox_to_mask(newInstanceMask,source_video_info.height,source_video_info.width)
        newInstance.mask = None
        newInstance.confidence = np.array(confidence[i],dtype='float32')
        newInstance.class_id = np.array(classId[i])
        newInstance.tracker_id = np.array(trackerId[i])
        newInstance.data = np.array(classIdByName[i],dtype='<U13')
        detections_arr.append(newInstance)

    trace_annotator = sv.TraceAnnotator(uniqueColorsFinal,color_lookup=ColorLookup.TRACK)
    bounding_box_annotator = sv.BoundingBoxAnnotator(uniqueColorsFinal,color_lookup=ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(uniqueColorsFinal,color_lookup=ColorLookup.TRACK)
    # print(newInstance.tracker_id)

    def callback2(frame: np.ndarray,detections,source_video_info) -> np.ndarray:
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]

        annotated_frame = frame.copy()

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

        return  annotated_frame

    print(source_video_info)
    with VideoSink(target_path=TARGET_VIDEO_PATH, video_info=source_video_info) as sink:
        for index, frame in enumerate(
            get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
        ):
            result_frame = callback2(frame,detections_arr[index],source_video_info)
            sink.write_frame(frame=result_frame)
            print("WritingFrame: "+str(index))
            if (index == numFrames):
                break

                
SOURCE_VIDEO_PATH = f"../videos/junctionEvalDataset/ChickPeaClips/clip0.mp4"
TARGET_VIDEO_PATH = f"deleteLater.mp4"

    # detections = loadDetections(filename=f"../longDetections.dat")
    # idDataset = cleaningDataset(detections=detections,frameThreshold=24)
    # obtainRectangles(SOURCE_VIDEO_PATH,finalDict=idDataset,numClusters=4)
# idDataset = fillMissingFrames(finalDict=idDataset,REGRESSION_POWER=4)
# writeToVideo(idDataset,detections=detections,SOURCE_VIDEO_PATH=SOURCE_VIDEO_PATH,TARGET_VIDEO_PATH=TARGET_VIDEO_PATH,numFrames=1000)