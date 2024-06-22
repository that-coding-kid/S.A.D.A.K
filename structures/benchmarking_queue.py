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
from utils.general import  load_zones_config
from locales.settings_languages import COMPONENTS

from structures.essentials import  save_to_csv, encrypt_it,make_headings, drop



KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
COLORS = sv.ColorPalette.DEFAULT

heading_made = False

def BenchMarking(source_path, zones_IN_configuration_path, weight_path, dataFrame: pandas.DataFrame, time_analysis:float, confidence: float,language:str, analysis_path:str):
    zone_dict = {}
    zone_id = {}
    in_map = {}
    FRAME_NUM = 1
    csv_list=[]
    iter = []
    print_iter={}
    iter_count = 1
    
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
            zone_id[zone] = dataFrame.columns[count]
            zone_dict[zone_id[zone]] = count
            in_map[zone_id[zone]] = count
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
        
        global heading_made
        annotated_frame = frame.copy()
        for zone, zone_annotator, box_annotator in zip(
            zones, zone_annotators, box_annotators
        ):
            detections_in_zone = detections[zone.trigger(detections=detections)]
            zone_dict[zone_id[zone]] = len(detections_in_zone)
            if(FRAME_NUM==4 or FRAME_NUM%(time_analysis*60*24)==0):
                iter_count = FRAME_NUM//(time_analysis*60*24) + 1
                if(print_iter.get(iter_count,0)==0):
                    st.subheader(str("Iteration: "+str(iter_count)))
                    print_iter[iter_count] = True
                
                zone_dict["Time"] = dataFrame.iloc[(FRAME_NUM//int(time_analysis*60*24))][0]
                st.write(str(zone_id[zone])+" : "+str(round(dataFrame.iloc[(FRAME_NUM//int(time_analysis*60*24))][in_map[zone_id[zone]]]/(zone_dict[zone_id[zone]]*3),2)))
                
                if(zone_dict not in csv_list):
                    csv_list.append(zone_dict)
                    
                
                if(not heading_made):
                    make_headings(analysis_path,csv_list)
                    save_to_csv(analysis_path,csv_list)
                    heading_made = True
                else:
                    save_to_csv(analysis_path,csv_list)
                    #encrypt_it(analysis_path)
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
    model = YOLO(weight_path)
    target = None
    vid_cap = cv2.VideoCapture(source_path)
    st_frame = st.empty()
    while(vid_cap.isOpened()):
        success = vid_cap.read()
        st.subheader(COMPONENTS[language]["ACCURACY"])
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
