from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import supervision as sv
import pandas
from tqdm import tqdm
import settings

import os
from typing import Any, Optional, Tuple, Dict, Iterable, List, Set
import shutil
import cv2
import numpy as np
from utils.general import find_in_list, load_zones_config

from scripts.jxnEvalDataCreation import mainFunc

from structures.VideoProcessor import VideoProcessor
from structures.essentials import drawzones
from structures.essentials import display_tracker_options, _display_detected_frames, load_model
from structures.encroachment import timedetect, livedetection


KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
COLORS = sv.ColorPalette.DEFAULT


class JunctionEvaluation:
    
    def __init__(self,sourcePath):
        self.sourcePath = sourcePath
        pass
    
    def datasetCreation(self,cycle):
        savePath = "videos/junctionEvalDataset/"
        print("ABC\n\n\n\n\n\n\n\n"+self.sourcePath)
        videoName = self.sourcePath[self.sourcePath.rfind("\\")+1:]
        videoName = videoName[:-4]
        finalpath = savePath+videoName+"Clips"
        isExist = os.path.exists(finalpath)
        if (isExist):
            shutil.rmtree(finalpath)
        os.makedirs(finalpath)
        mainFunc(self.sourcePath,cycle,finalpath)
        settings.updateDirectories()
        return finalpath
            
        
def startup():
    settings.updateDirectories()

def play_youtube_video(conf, model, language):

    source_youtube = st.sidebar.text_input(settings.COMPONENTS[language]["YOUTUBEURL"])

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button(settings.COMPONENTS[language]["DETECT_OBJ"]):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(settings.COMPONENTS[language]["VIDEO_ERROR"] + str(e))

def play_stored_video(conf, model,language):

    source_vid = st.sidebar.selectbox(
        settings.COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT[source_vid], 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button(settings.COMPONENTS[language]["DETECT_OBJ"]):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error(settings.COMPONENTS[language]["VIDEO_ERROR"] + str(e))



def play_rtsp_stream(conf, model, language):
    source_rtsp = st.sidebar.text_input(settings.COMPONENTS[language]["RTSPSTREAM"])
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button(settings.COMPONENTS[language]["DETECT_OBJ"]):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error(settings.COMPONENTS[language]["RTSP_ERROR"] + str(e))


def enchroachment(confidence: float, language: str):
    source_vid = st.sidebar.selectbox(
    settings.COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))
    time = st.sidebar.text_input(settings.COMPONENTS[language]["VIOLATION_TIME"])
    source_url = st.sidebar.text_input(settings.COMPONENTS[language]["SOURCE_URL_RTSP"])
    cwd = os.getcwd()
    if st.sidebar.button(settings.COMPONENTS[language]["BOTTLENECK_ERRORS"]):
        if(source_url): 
            zones_configuration_path = os.path.join(cwd,zones_configuration_path)
            livedetection(source_url=source_url, violation_time=int(time), zone_configuration_path=zones_configuration_path)
        else:
            new_path = source_path.split("\\")[-1]
            zones_configuration_path = "configure\ZONES"+new_path+".json" 
            if(os.path.exists(zones_configuration_path)):
                timedetect(source_path = source_path, zone_configuration_path = zones_configuration_path, violation_time=float(time))
            else:
                drawzones(source_path = source_path, zone_configuration_path = zones_configuration_path)
                timedetect(source_path = source_path, zone_configuration_path = zones_configuration_path, violation_time=float(time), confidence=confidence)
                
def junctionEvaluationDataset(language: str):
    source_vid = st.sidebar.selectbox(
    settings.COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))

    successVar = False
    cycle = []
    try:
        cycle = st.sidebar.text_input(settings.COMPONENTS[language]["CYCLE"])
        cycle = cycle.split()
        cycle = [int (i) for i in cycle]
        successVar = True
    except:
        pass
    # time = st.sidebar.text_input("Violation Time:")
    #source_url = st.sidebar.text_input("Source Url:")
    
    if st.sidebar.button(settings.COMPONENTS[language]["CREATE_DATASET"]):
        if (successVar == False):
            st.sidebar.error(settings.COMPONENTS[language]["INVALID_SYNTAX"])
            pass
        else:
            jxnEvalInstance = JunctionEvaluation(source_path)
            returnPath = jxnEvalInstance.datasetCreation(cycle=cycle)
            st.sidebar.write(settings.COMPONENTS[language]["SUCCESS_DATA"]+returnPath)
                  
def junctionEvaluation(language):
    if (len(settings.EVALUATION_DICT.keys()) == 0):
        st.sidebar.error(settings.COMPONENTS[language]["DATASET_NOT_THERE"])
    else:
        source_dir = st.sidebar.selectbox(
        settings.COMPONENTS[language]["CHOOSE_FOLDER"], settings.EVALUATION_DICT.keys())
        
        source_path = str(settings.EVALUATION_DICT.get(source_dir))
        source_vid = st.sidebar.selectbox(
        settings.COMPONENTS[language]["CHOOSE_VID"], settings.FINAL_DICT[source_dir].keys())
        
        
        with open("videos/JunctionEvalDataset/"+source_dir+"/"+source_vid, 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)

        threshold = st.sidebar.text_input(
            settings.COMPONENTS[language]["INTEGER_RANGE"]
        )

        try:
            
            threshold = int(threshold)
            if (threshold > 5 or threshold < 1):
                st.sidebar.error(settings.COMPONENTS[language]["VALID_VALUE"])
            else:
                if st.sidebar.button(settings.COMPONENTS[language]["EVALUATION"]):
                    returnVid = "videos/JunctionEvaluations/IndiraNagarClips/clip1.mp4"
                    with open(returnVid, 'rb') as video_file2:
                        video_bytes2 = video_file2.read()
                        
                    if video_bytes2:
                        st.video(video_bytes2)
                    
                                                            
        except:
            st.sidebar.error(settings.COMPONENTS[language]["VALID_VALUE"])            


def benchMarking(confidence: float, language:str):
    source_vid = st.sidebar.selectbox(
    settings.COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))
    traffic_data = st.sidebar.file_uploader(settings.COMPONENTS[language]["TRAFFIC_DATA"], type=("csv"))
    time = st.sidebar.text_input(settings.COMPONENTS[language]["ACCURACY_INTERVAL"])
    choice = st.sidebar.radio(settings.COMPONENTS[language]["BENCHMARKING_CRIT"], [settings.COMPONENTS[language]["BENCHMARKING_FLOW"], settings.COMPONENTS[language]["BENCHMARKING_QUEUE_LENGTH"]])
    new_path = source_path.split("\\")[-1]
    zones_IN_configuration_path = "configure/ZONES_IN"+new_path+".json"
    zones_OUT_configuration_path = "configure/ZONES_OUT"+new_path+".json"
    weight_path = "weights/yolov8n.pt"
    if(st.sidebar.button(settings.COMPONENTS[language]["ZONES_IN"])):
        drawzones(source_path = source_path, zone_configuration_path = zones_IN_configuration_path)
        st.sidebar.write("ZONES_IN "+settings.COMPONENTS[language]["SUCCESS"]+zones_IN_configuration_path)
    
    if(st.sidebar.button(settings.COMPONENTS[language]["ZONES_OUT"])):    
        drawzones(source_path = source_path, zone_configuration_path = zones_OUT_configuration_path)
        st.sidebar.write("ZONES_OUT "+settings.COMPONENTS[language]["SUCCESS"]+zones_OUT_configuration_path)
    if(st.sidebar.button(settings.COMPONENTS[language]["BENCHMARK"])):
            if traffic_data is not None:
                df = pandas.read_csv(traffic_data)
                if(choice == settings.COMPONENTS[language]["BENCHMARKING_FLOW"]):
                    processor = VideoProcessor(
                    source_weights_path=weight_path,
                    source_video_path=source_path,
                    zoneIN_configuration_path=zones_IN_configuration_path,
                    zoneOUT_configuration_path=zones_OUT_configuration_path,  
                    time = float(time),
                    confidence_threshold=confidence
                )
                    processor.process_video()
                elif settings.COMPONENTS[language]["BENCHMARKING_QUEUE_LENGTH"]:
                    BenchMarking(source_path=source_path, zones_IN_configuration_path=zones_IN_configuration_path, weight_path=weight_path, dataFrame=df, time_analysis=float(time),confidence=confidence, language=language)
            else:
                st.sidebar.warning(settings.COMPONENTS[language]["TRAFFIC_DATA_NOT_UPLOADED"])
                return
            
                

def BenchMarking(source_path, zones_IN_configuration_path, weight_path, dataFrame: pandas.DataFrame, time_analysis:float, confidence: float,language:str):
    zone_dict = {}
    zone_id = {}
    FRAME_NUM = 1
    def initiate_annotators(
    polygons: List[np.ndarray], resolution_wh: Tuple[int, int]
) -> Tuple[
    List[sv.PolygonZone], List[sv.PolygonZoneAnnotator], List[sv.BoundingBoxAnnotator]
]:
        count = 1
        line_thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        zones = []
        zone_annotators = []
        box_annotators = []
        
        for index, polygon in enumerate(polygons):
            zone = sv.PolygonZone(polygon=polygon)
            zone_annotator = sv.PolygonZoneAnnotator(
                zone=zone,
                color=COLORS.by_idx(index),
                thickness=line_thickness,
                text_thickness=line_thickness * 2,
                text_scale=text_scale * 2,
            )
            box_annotator = sv.BoundingBoxAnnotator(
                color=COLORS.by_idx(index), thickness=line_thickness
            )
            zones.append(zone)
            zone_id[zone] = count
            zone_dict[zone] = count
            zone_annotators.append(zone_annotator)
            box_annotators.append(box_annotator)
            count+=1

        return zones, zone_annotators, box_annotators

    def detect(
        frame: np.ndarray, model: YOLO, confidence_threshold: float = 0.5
    ) -> sv.Detections:
    
        results = model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        return detections


    def annotate(
        frame: np.ndarray,
        zones: List[sv.PolygonZone],
        zone_annotators: List[sv.PolygonZoneAnnotator],
        box_annotators: List[sv.BoundingBoxAnnotator],
        detections: sv.Detections,
        FRAME_NUM: int
    ) -> np.ndarray:
        
        
        annotated_frame = frame.copy()
        for zone, zone_annotator, box_annotator in zip(
            zones, zone_annotators, box_annotators
        ):
            detections_in_zone = detections[zone.trigger(detections=detections)]
            zone_dict[zone] = len(detections_in_zone)
            if(FRAME_NUM==4 or FRAME_NUM%(time_analysis*60*24)==0):
                zone_dict[zone] = dataFrame.iloc[(FRAME_NUM//int(time_analysis*60*24))][zone_id[zone]]/(zone_dict[zone]*3)
                st.write(str(str(zone_id[zone])+" : "+str(zone_dict[zone])))
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections_in_zone
            )
        return annotated_frame

    
    video_info = sv.VideoInfo.from_video_path(source_path)
    polygons = load_zones_config(zones_IN_configuration_path)
    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=video_info.resolution_wh
    )
    print(zone_dict)
    print(zone_id)
    model = YOLO(weight_path)
    target = None
    vid_cap = cv2.VideoCapture(source_path)
    st_frame = st.empty()
    while(vid_cap.isOpened()):
        success = vid_cap.read()
        st.subheader(settings.COMPONENTS[language]["ACCURACY"])
        if success:
            frames_generator = sv.get_video_frames_generator(source_path)
            if  target is not None:
                with sv.VideoSink(target, video_info) as sink:
                    for frame in tqdm(frames_generator, total=video_info.total_frames):
                        detections = detect(frame, model, confidence_threshold=confidence)
                        annotated_frame = annotate(
                            frame=frame,
                            zones=zones,
                            zone_annotators=zone_annotators,
                            box_annotators=box_annotators,
                            detections=detections,
                            FRAME_NUM=FRAME_NUM
                        )
                        
                        sink.write_frame(annotated_frame)
            else:
                for frame in tqdm(frames_generator, total=video_info.total_frames):
                    detections = detect(frame, model, confidence_threshold=confidence)
                    annotated_frame = annotate(
                        frame=frame,
                        zones=zones,
                        zone_annotators=zone_annotators,
                        box_annotators=box_annotators,
                        detections=detections,
                        FRAME_NUM=FRAME_NUM
                    )
                    FRAME_NUM+=1
                    
                    st_frame.image(annotated_frame,
                                caption='Detected Video',
                                channels="BGR",
                                use_column_width=True)
                vid_cap.release
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.destroyAllWindows()
