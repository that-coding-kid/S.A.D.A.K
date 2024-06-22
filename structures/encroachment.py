from ultralytics import YOLO

import streamlit as st
import cv2

import supervision as sv


import settings
from locales.settings_languages import COMPONENTS
from typing import Any, Optional, Tuple, Dict, Iterable, List, Set

import cv2
import numpy as np
from inference import get_roboflow_model
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
from inference import InferencePipeline
from structures.essentials import save_to_csv, encrypt_it,make_headings, drop, send_ping
from structures.CustomSink import CustomSink
KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
FRAME_NUM = 0
THICKNESS = 2
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]
violations = []
displayed={}
headings_made = False
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)
current_mouse_position: Optional[Tuple[int, int]] = None


def timedetect(source_path, zone_configuration_path, violation_time, confidence, language: str, analysis_path: str):
    COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
    COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
    LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
    )
    
    if "terminate" not in st.session_state:
        st.session_state["terminate"] = False
    csv_list =[]
    alert_dicts = {}
    time_in_seconds = False
    time_in_minutes = False
    if(violation_time>=1):
        time_in_minutes = True
    else:
        time_in_seconds = True
    model_id = "custom"
    classes = [2,5,6,7,121]
    iou = 0.7
    model = YOLO("weights/best_ps3.pt")
    tracker = sv.ByteTrack(minimum_matching_threshold=confidence)
    video_info = sv.VideoInfo.from_video_path(video_path=source_path)
    frames_generator = sv.get_video_frames_generator(source_path)
    global headings_made
    polygons = load_zones_config(file_path=zone_configuration_path)
    
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]
                                
    vid_cap = cv2.VideoCapture(source_path)
    st_frame = st.empty()
    
    while(vid_cap.isOpened()):
         if(not st.session_state["terminate"]):
                terminate = st.button("Terminate")
                success = vid_cap.read()
                st.subheader(COMPONENTS[language]["ALERTS"])
                if success:
                        
                        for frame in frames_generator:
                            
                            results = model(frame, verbose=False, device="cpu", conf=confidence)[0]
                            detections = sv.Detections.from_ultralytics(results)
                            detections = detections[find_in_list(detections.class_id, classes)]
                            detections = detections.with_nms(threshold=iou)
                            detections = tracker.update_with_detections(detections)

                            annotated_frame = frame.copy()
                            
                            for idx, zone in enumerate(zones):
                                annotated_frame = sv.draw_polygon(
                                    scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                                )

                                detections_in_zone = detections[zone.trigger(detections)]
                                time_in_zone = timers[idx].tick(detections_in_zone)
                                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                                annotated_frame = COLOR_ANNOTATOR.annotate(
                                    scene=annotated_frame,
                                    detections=detections_in_zone,
                                    custom_color_lookup=custom_color_lookup,
                                )
                                labels = [
                                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)     
                                ]

                                annotated_frame = LABEL_ANNOTATOR.annotate(
                                    scene=annotated_frame,
                                    detections=detections_in_zone,
                                    labels=labels,
                                    custom_color_lookup=custom_color_lookup,
                                )
                                
                                for tracker_ID, time, cl in zip(detections_in_zone.tracker_id, time_in_zone, detections_in_zone.class_id):
                                    if tracker_ID not in displayed:
                                        if(time_in_minutes):
                                            if(time//60 >= float(violation_time)):
                                                done_once = False
                                                violations.append(tracker_ID)
                                                cla = settings.CLASSES[cl]
                                                alert_dicts["Tracker_ID"] = str(tracker_ID)
                                                alert_dicts["Class"] = cla
                                                alert_dicts["Location"] = idx
                                                s = "Tracker_ID:" + str(tracker_ID) + " Class: " + cla + " Location: "+str(idx)
                                                st.warning(s, icon= "⚠️")
                                                send_ping(email="mangalsaativk@gmail.com",file=s)
                                                displayed[tracker_ID] = 1
                                                if(alert_dicts not in csv_list):                                            
                                                    csv_list.append(alert_dicts)
                                                if(not headings_made):
                                                    make_headings(analysis_path,csv_list)
                                                    save_to_csv(analysis_path,csv_list)
                                                    headings_made = True
                                                else:
                                                    if(not done_once):
                                                        save_to_csv(analysis_path,csv_list)
                                                        drop(analysis_path,csv_list)
                                                        #encrypt_it(analysis_path)
                                                        done_once = True
                                                        continue
                                                    
                                                
                                        if(time_in_seconds):
                                            if(time%60 >= float(violation_time*60)):
                                                violations.append(tracker_ID)
                                                done_once = False
                                                cla = settings.CLASSES[cl]
                                                alert_dicts["Tracker_ID"] = str(tracker_ID)
                                                alert_dicts["Class"] = cla
                                                alert_dicts["Location"] = str(idx)
                                                s = "Tracker_ID:" + str(tracker_ID) + " Class: " + cla + " Location: "+str(idx)
                                                st.warning(s, icon= "⚠️")
                                                send_ping(email="mangalsaatvik@gmail.com",file=s)
                                                displayed[tracker_ID] = 1  
                                                if(alert_dicts not in csv_list):                                          
                                                    csv_list.append(alert_dicts)
                                                
                                                if(not headings_made):
                                                    make_headings(analysis_path,csv_list)
                                                    save_to_csv(analysis_path,csv_list)
                                                    headings_made = True
                                                else:
                                                    if(not done_once):
                                                        save_to_csv(analysis_path,csv_list)
                                                        #encrypt_it(analysis_path)
                                                        done_once = True
                                                        continue


                            st_frame.image(annotated_frame,
                                    caption='Detected Video',
                                    channels="BGR",
                                    use_column_width=True)
    if(st.session_state["terminate"]):
        st.success("Stream Terminated")
        st.session_state["terminate"] = False


def livedetection(source_url: str, violation_time: int, zone_configuration_path: str, confidence: float, analysis_path: str):
    model_id = 'yolov8x-640'
    classes = [2,5,6,7]
    frame = st.empty()
    iou = 0.7
    model = YOLO('weights\yolov8n.pt')
    terminate = st.button("Terminate Stream")
    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes, violation_time = violation_time, frame=frame, analysis_path = analysis_path)

    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=source_url,
        on_prediction=sink.on_prediction,
        confidence=confidence,
        iou_threshold=iou,
    )
    

    pipeline.start()

    try:
        pipeline.join()
    except terminate:
        pipeline.terminate()

