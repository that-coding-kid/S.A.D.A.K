import streamlit as st
import cv2
import supervision as sv
from typing import Any, Optional, Tuple, Dict, Iterable, List, Set
import cv2
import numpy as np
from utils.general import find_in_list, load_zones_config
from inference.core.interfaces.camera.entities import VideoFrame
from utils.timers import ClockBasedTimer
import settings
from structures.essentials import save_to_csv,make_headings,encrypt_it
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
alert_dicts ={}
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)
current_mouse_position: Optional[Tuple[int, int]] = None
csv_list = []
headings_made = False



class CustomSink:
    def __init__(self, zone_configuration_path: str, classes, violation_time: int, frame, analysis_path:str):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]
        self.violation_time = violation_time
        self.frame = frame
        self.csv = csv_list
        self.analysis_path = analysis_path

    def on_prediction(self, result: dict, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps
        global headings_made
        detections = sv.Detections.from_inference(result)
        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)
        alert_dicts = {}
        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )
        if(self.violation_time>=1):
            time_in_minutes = True
        else:
            time_in_seconds = True

        for idx, zone in enumerate(self.zones):
            
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = self.timers[idx].tick(detections_in_zone)
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
            for tracker_ID, time, cl in zip(detections_in_zone.tracker_id, time_in_zone, detections_in_zone.class_id):
                if tracker_ID not in displayed:
                    if(time_in_minutes):
                        if(time//60 >= float(self.violation_time)):
                            violations.append(tracker_ID)
                            cla = settings.CLASSES[cl]
                            alert_dicts["Tracker_ID"] = str(tracker_ID)
                            alert_dicts["Class"] = cla
                            alert_dicts["Location"] = str(idx)
                            if(alert_dicts not in csv_list):
                                csv_list.append(alert_dicts)
                            if(not headings_made):
                                make_headings(self.analysis_path,csv_list)
                                save_to_csv(self.analysis_path,csv_list)
                                headings_made = True
                            else:
                                save_to_csv(self.analysis_path,csv_list)
                                encrypt_it(self.analysis_path)
                            s = "Tracker_ID:" + str(tracker_ID) + " Class: " + cla + " Location: "+str(idx)
                            st.warning(s, icon= "⚠️")
                            displayed[tracker_ID] = 1
                            
                    if(time_in_seconds):
                        if(time%60 >= float(self.violation_time*60)):
                            violations.append(tracker_ID)
                            cla = settings.CLASSES[cl]
                            alert_dicts["Tracker_ID"] = str(tracker_ID)
                            alert_dicts["Class"] = cla
                            alert_dicts["Location"] = str(idx)
                            if(alert_dicts not in csv_list):
                                csv_list.append(alert_dicts)
                            if(not headings_made):
                                make_headings(self.analysis_path,csv_list)
                                save_to_csv(self.analysis_path,csv_list)
                                headings_made = True
                            else:
                                save_to_csv(self.analysis_path,csv_list)
                                encrypt_it(self.analysis_path)                            
                            s = "Tracker_ID:" + str(tracker_ID) + " Class: " + cla + " Location: " + str(idx)
                            st.warning(s, icon= "⚠️")
                            displayed[tracker_ID] = 1
                
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )
        self.frame.image(annotated_frame,
                                   caption='Detected Video',
                                   channels="BGR",
                                   use_column_width=True)
        cv2.waitKey(1)
