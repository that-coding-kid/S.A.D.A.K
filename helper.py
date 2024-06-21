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
from locales.settings_languages import COMPONENTS
from scripts.jxnEvalDataCreation import mainFunc
from structures.VideoProcessor import VideoProcessor
from structures.essentials import drawzones, decrypt_it, encrypt_it
from structures.essentials import display_tracker_options, _display_detected_frames, load_model, get_first_frame
from structures.encroachment import timedetect, livedetection
from structures.benchmarking_queue import BenchMarking
from PIL import Image
from scripts.jxnEvaluator import *
import plotly.express as px
import plotly.figure_factory as ff

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
COLORS = sv.ColorPalette.DEFAULT
CARD_IMAGE_SIZE = (300, int(300*0.5625))
VIDEO_DIR_PATH = f"videos/"
IMAGES_DIR_PATH = f"images/"
DETECTIONS_DIR_PATH = f"detections/"


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

    source_youtube = st.sidebar.text_input(COMPONENTS[language]["YOUTUBE_URL"])

    is_display_tracker, tracker = display_tracker_options(language=language)

    if st.sidebar.button(COMPONENTS[language]["DETECT_OBJ"]):
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
            st.sidebar.error(COMPONENTS[language]["VIDEO_ERROR"] + str(e))

def play_stored_video(conf, model,language):

    source_vid = st.sidebar.selectbox(
        COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options(language=language)

    with open(settings.VIDEOS_DICT[source_vid], 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button(COMPONENTS[language]["DETECT_OBJ"]):
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
            st.sidebar.error(COMPONENTS[language]["VIDEO_ERROR"] + str(e))

def play_rtsp_stream(conf, model, language):
    source_rtsp = st.sidebar.text_input(COMPONENTS[language]["RTSPSTREAM"])
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options(language=language)
    if st.sidebar.button(COMPONENTS[language]["DETECT_OBJ"]):
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
            st.sidebar.error(COMPONENTS[language]["RTSP_ERROR"] + str(e))

def enchroachment(confidence: float, language: str):
    global CURRENT_DIR_PATH

    if ("current_dir_path" not in st.session_state):
        st.session_state.current_dir_path = VIDEO_DIR_PATH

    st.text(st.session_state.current_dir_path)
    if ("current_state" not in st.session_state):
        st.session_state.current_state = "homePage"



    if (st.session_state.current_state=="homePage"):
        st.header("Welcome to Bottleneck Detection")
        isVideo = False
        if (st.session_state.current_dir_path.endswith(('.mp4', '.avi','.mov','.AVI'))):
            isVideo = True
            st.session_state.current_state = "Encroachment"
            st.rerun()
        else:
            st.title("Video Gallery")
        
        
        video_files = [f for f in os.listdir(st.session_state.current_dir_path) if f.endswith(('.mp4', '.avi','.mov','.AVI'))]
        folders = [f for f in os.listdir(st.session_state.current_dir_path) if '.' not in f]
        # Display videos in a grid
        cols = st.columns(3)  # Adjust the number of columns as needed
        video_files = folders+video_files
        if (st.session_state.current_dir_path!=VIDEO_DIR_PATH):
            if (st.button("Back")):
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind("/")]
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind("/")+1]
                st.rerun()

        for idx, video_file in enumerate(video_files):
            with cols[idx % 3]:  # Change 3 to the number of columns you want
                video_path = os.path.join(st.session_state.current_dir_path, video_file)
                first_frame = get_first_frame(video_path)
                if (idx < len(folders)):
                    first_frame = Image.open(IMAGES_DIR_PATH+"/FolderIcon.png")
                    first_frame = first_frame.resize(CARD_IMAGE_SIZE, Image.LANCZOS)
                    st.image(first_frame, use_column_width=True)
                    st.write(video_file)
                    if st.button(f"Navigate to {video_file}", key=video_file):
                        st.session_state.current_dir_path= st.session_state.current_dir_path+video_file+"/"
                        st.rerun()
                        
                else:
                    if first_frame:
                        st.image(first_frame, use_column_width=True)
                        st.write(video_file)
                        if st.button(f"Detect Bottlenecks for {video_file}", key=video_file):
                            st.session_state.current_dir_path = st.session_state.current_dir_path+video_file
                            st.rerun()

    if(st.session_state.current_state == "Encroachment"):                
    #source_vid = st.sidebar.selectbox(
    #COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())
        
        source_path = st.session_state.current_dir_path
        col1, col2 = st.columns([3,1])
        with col1:
            st.title("Bottleneck Detection: " + st.session_state.current_dir_path[st.session_state.current_dir_path.rfind("/")+1:])
        with col2:
            if (st.button("Back to video gallery")):
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind('/')+1]
                st.session_state.current_state = "homePage"
                st.rerun()
        time = st.text_input(COMPONENTS[language]["VIOLATION_TIME"], placeholder="Enter Violation Time")
        
        source_url = st.text_input(COMPONENTS[language]["SOURCE_URL_RTSP"], placeholder="Enter RTSP URL:")
        if st.button(COMPONENTS[language]["BOTTLENECK_ERRORS"]):
            if(source_url): 
                zones_configuration_path = "configure/ZONESFootage_Feed_6.mp4.json"
                analysis_path = "analysis/encroachments/data_"+source_url+".csv"
                livedetection(source_url=source_url, violation_time=int(time), zone_configuration_path=zones_configuration_path,confidence=confidence, analysis_path=analysis_path)
            else:
                new_path = source_path.split("/")[-1]
                zones_configuration_path = "configure/ZONES"+new_path+".json" 
                
                analysis_path = "analysis/encroachments/data_encroachment"+new_path+".csv"
                if(os.path.exists(zones_configuration_path)):
                    timedetect(source_path = source_path, zone_configuration_path = zones_configuration_path, violation_time=float(time), confidence=confidence, language=language, analysis_path=analysis_path)
                                

                else:
                    drawzones(source_path = source_path, zone_configuration_path = zones_configuration_path)
                    timedetect(source_path = source_path, zone_configuration_path = zones_configuration_path, violation_time=float(time), confidence=confidence, language=language, analysis_path=analysis_path) #Removed csvList arguement 
                                   
def junctionEvaluationDataset(language: str):
    source_vid = st.sidebar.selectbox(
    COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())
    source_path = str(settings.VIDEOS_DICT.get(source_vid))

    successVar = False
    cycle = []
    try:
        cycle = st.sidebar.text_input(COMPONENTS[language]["CYCLE"])
        cycle = cycle.split()
        cycle = [int (i) for i in cycle]
        successVar = True
    except:
        pass
    # time = st.sidebar.text_input("Violation Time:")
    #source_url = st.sidebar.text_input("Source Url:")
    
    if st.sidebar.button(COMPONENTS[language]["CREATE_DATASET"]):
        if (successVar == False):
            st.sidebar.error(COMPONENTS[language]["INVALID_SYNTAX"])
            pass
        else:
            jxnEvalInstance = JunctionEvaluation(source_path)
            returnPath = jxnEvalInstance.datasetCreation(cycle=cycle)
            st.sidebar.write(COMPONENTS[language]["SUCCESS_DATA"]+returnPath)
                  
# def junctionEvaluation(language):
#     if (len(settings.EVALUATION_DICT.keys()) == 0):
#         st.sidebar.error(COMPONENTS[language]["DATASET_NOT_THERE"])
#     else:
#         source_dir = st.sidebar.selectbox(
#         COMPONENTS[language]["CHOOSE_FOLDER"], settings.EVALUATION_DICT.keys())
        
#         source_path = str(settings.EVALUATION_DICT.get(source_dir))
#         source_vid = st.sidebar.selectbox(
#         COMPONENTS[language]["CHOOSE_VID"], settings.FINAL_DICT[source_dir].keys())
        
        
#         with open("videos/JunctionEvalDataset/"+source_dir+"/"+source_vid, 'rb') as video_file:
#             video_bytes = video_file.read()
#         if video_bytes:
#             st.video(video_bytes)

#         threshold = st.sidebar.text_input(
#             COMPONENTS[language]["INTEGER_RANGE"]
#         )

#         try:
            
#             threshold = int(threshold)
#             if (threshold > 5 or threshold < 1):
#                 st.sidebar.error(COMPONENTS[language]["VALID_VALUE"])
#             else:
#                 if st.sidebar.button(COMPONENTS[language]["EVALUATION"]):
#                     returnVid = "videos/JunctionEvaluations/IndiraNagarClips/clip1.mp4"
#                     with open(returnVid, 'rb') as video_file2:
#                         video_bytes2 = video_file2.read()
                        
#                     if video_bytes2:
#                         st.video(video_bytes2)
                    
                                                            
#         except:
#             st.sidebar.error(COMPONENTS[language]["VALID_VALUE"])            

def benchMarking(confidence: float, language:str):
    
    global CURRENT_DIR_PATH

    if ("current_dir_path" not in st.session_state):
        st.session_state.current_dir_path = VIDEO_DIR_PATH

    st.text(st.session_state.current_dir_path)
    if ("current_state" not in st.session_state):
        st.session_state.current_state = "homePage"



    if (st.session_state.current_state=="homePage"):
        st.header("Welcome To Benchmarking")
        isVideo = False
        if (st.session_state.current_dir_path.endswith(('.mp4', '.avi','.mov','.AVI'))):
            isVideo = True
            st.session_state.current_state = "Benchmarking"
            st.rerun()
        else:
            st.title("Video Gallery")
        
        
        video_files = [f for f in os.listdir(st.session_state.current_dir_path) if f.endswith(('.mp4', '.avi','.mov','.AVI'))]
        folders = [f for f in os.listdir(st.session_state.current_dir_path) if '.' not in f]
        # Display videos in a grid
        cols = st.columns(3)  # Adjust the number of columns as needed
        video_files = folders+video_files
        if (st.session_state.current_dir_path!=VIDEO_DIR_PATH):
            if (st.button("Back")):
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind("/")]
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind("/")+1]
                st.rerun()

        for idx, video_file in enumerate(video_files):
            with cols[idx % 3]:  # Change 3 to the number of columns you want
                video_path = os.path.join(st.session_state.current_dir_path, video_file)
                first_frame = get_first_frame(video_path)
                if (idx < len(folders)):
                    first_frame = Image.open(IMAGES_DIR_PATH+"/FolderIcon.png")
                    first_frame = first_frame.resize(CARD_IMAGE_SIZE, Image.LANCZOS)
                    st.image(first_frame, use_column_width=True)
                    st.write(video_file)
                    if st.button(f"Navigate to {video_file}", key=video_file):
                        st.session_state.current_dir_path= st.session_state.current_dir_path+video_file+"/"
                        st.rerun()
                        
                else:
                    if first_frame:
                        st.image(first_frame, use_column_width=True)
                        st.write(video_file)
                        if st.button(f"Benchmark {video_file}", key=video_file):
                            st.session_state.current_dir_path = st.session_state.current_dir_path+video_file
                            st.rerun()
    
    if(st.session_state.current_state == "Benchmarking"):
    
        #source_vid = st.sidebar.selectbox(
        #COMPONENTS[language]["CHOOSE_VID"], settings.VIDEOS_DICT.keys())
        
        
        col1, col2 = st.columns([3,1])
        with col1:
            st.title("Benchmark: " + st.session_state.current_dir_path[st.session_state.current_dir_path.rfind("/")+1:])
        with col2:
            if (st.button("Back to video gallery")):
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind('/')+1]
                st.session_state.current_state = "homePage"
                st.rerun()
        
        source_path = st.session_state.current_dir_path
        traffic_data = st.file_uploader(COMPONENTS[language]["TRAFFIC_DATA"], type=("csv"))
        
        time = st.text_input(COMPONENTS[language]["ACCURACY_INTERVAL"])
        choice = st.radio(COMPONENTS[language]["BENCHMARKING_CRIT"], [COMPONENTS[language]["BENCHMARKING_FLOW"], COMPONENTS[language]["BENCHMARKING_QUEUE_LENGTH"]])
        
        new_path = source_path.split("/")[-1]
        
        zones_IN_configuration_path = "configure/ZONES_IN"+new_path+".json"
        zones_OUT_configuration_path = "configure/ZONES_OUT"+new_path+".json"
        weight_path = "weights/yolov8n.pt"
        col1, col2, col3 = st.columns(3)
        
        with col3:
            button1 = st.button(COMPONENTS[language]["ZONES_IN"])

        with col2:
            button2 = st.button(COMPONENTS[language]["ZONES_OUT"])

        with col1:
            button3 = st.button(COMPONENTS[language]["BENCHMARK"])
        
        if(button1):
            drawzones(source_path = source_path, zone_configuration_path = zones_IN_configuration_path)
            st.sidebar.write("ZONES_IN "+COMPONENTS[language]["SUCCESS"]+zones_IN_configuration_path)
        
        if(button2):    
            drawzones(source_path = source_path, zone_configuration_path = zones_OUT_configuration_path)
            st.sidebar.write("ZONES_OUT "+COMPONENTS[language]["SUCCESS"]+zones_OUT_configuration_path)
            
        if(button3):
            
                if traffic_data is not None:
                    df = pandas.read_csv(traffic_data)
                    if(choice == COMPONENTS[language]["BENCHMARKING_FLOW"]):
                        analysis_path = "analysis\\accuracy\Flow Rate\data_flow_rate"+new_path+".csv"
                        processor = VideoProcessor(
                        source_weights_path=weight_path,
                        source_video_path=source_path,
                        zoneIN_configuration_path=zones_IN_configuration_path,
                        zoneOUT_configuration_path=zones_OUT_configuration_path,  
                        time = float(time),
                        confidence_threshold=confidence,
                        dataFrame = df,
                        analysis_path = analysis_path
                    )
                        processor.process_video()
                    elif COMPONENTS[language]["BENCHMARKING_QUEUE_LENGTH"]:
                        analysis_path = "analysis\\accuracy\Queue Length\data_queuelength"+new_path+".csv"
                        BenchMarking(source_path=source_path, zones_IN_configuration_path=zones_IN_configuration_path, weight_path=weight_path, dataFrame=df, time_analysis=float(time),confidence=confidence, language=language, analysis_path=analysis_path)
                else:
                    st.sidebar.warning(COMPONENTS[language]["TRAFFIC_DATA_NOT_UPLOADED"])
                    return
            
def Analyze(language):
    global showGraph
    if "authorize" not in st.session_state:
        st.session_state.authorize = "LogOut"
    if "current" not in st.session_state:
        st.session_state.current = "Home"
    
    if st.session_state.current == "Home":
        st.header("Welcome To Analysis")
        st.subheader("Authorization: ")
        auth_token=st.text_input("Auth Token",type="password", placeholder="Enter Your authorization token")
        auth_button = st.button("LOG IN")
        if(auth_button):
            if(auth_token == settings.ENCRYPTION_KEY):
                st.session_state.current = "Login" 
                st.session_state.authorize = "Login"
                st.rerun()
            else:
                st.session_state.current = "Home"
                st.sidebar.error("Invalid Auth Token!")
                st.rerun()
                
    if st.session_state.authorize == "Login":
        
        logout = st.sidebar.button("Log Out")
        if(logout):
            st.session_state.current = "Home"
            st.session_state.authorize = "LogOut"
            st.success("Logged Out")
            st.rerun()
     
        analysis_crit = st.selectbox("Analysis Criteria",
            [COMPONENTS[language]["ACCURACY"],COMPONENTS[language]["ENCROACHMENT"],"Encryption"]
        )
        
        st.session_state.current = analysis_crit
        
    if(st.session_state.current!="Encryptions" and st.session_state.authorize == "Login"):

        visualize = st.button("Visualize")
        if(visualize):
            st.session_state.current = "Visualize"
        
        if(st.session_state.current == COMPONENTS[language]["ACCURACY"]):
            
            if("parameter" not in st.session_state):
                st.session_state.parameter = "Queue Length" 
            parameter = st.radio("Choose Parameter",[COMPONENTS[language]["FLOW_HEADER"].split(":")[-2],"Queue Length"])
            st.session_state.parameter = parameter
            
            if st.session_state.parameter == COMPONENTS[language]["FLOW_HEADER"].split(":")[-2]:
                files = st.selectbox("Select file for analysis",settings.FLOW_DICT.keys())
                file_path = str(settings.FLOW_DICT.get(files))
                
                execute_1 = st.button("Begin Analyis")
                if(execute_1):
                    try:
                        decrypt_it(file_path, key = auth_token)
                        df = pandas.read_csv(file_path)
                        st.write(df)
                    except: 
                        df = pandas.read_csv(file_path, on_bad_lines='skip')
                        st.write(df)
                    if(showGraph):
                        fig = px.scatter(df, x="Time", y=list(df.columns)[1:], width = 800, height = 400)
                        fig.update_layout(showlegend = False)
                        st.plotly_chart(fig)
                        showGraph = False
             
            if st.session_state.parameter == "Queue Length":
                files = st.selectbox("Select file for analysis",settings.QUEUE_DICT.keys())
                file_path = str(settings.QUEUE_DICT.get(files))
    
                execute_1 = st.button("Begin Analyis")
                if(execute_1):
                    try:
                        decrypt_it(file_path, key = auth_token)
                        df = pandas.read_csv(file_path)
                        df = df.drop_duplicates(subset=["Time"])
                        st.write(df)
                    except: 
                        df = pandas.read_csv(file_path)
                        df = df.drop_duplicates(subset=["Time"])
                        st.write(df)
                    if(showGraph):
                        fig = px.scatter(df, x="Time", y=list(df.columns)[1:], width = 800, height = 400)
                        fig.update_layout(showlegend = False)
                        st.plotly_chart(fig)
                        showGraph = False
                    
                    #st.write(decrypt_it(file_path, key = auth_token))
            
        if(st.session_state.current == COMPONENTS[language]["ENCROACHMENT"]):
            files = st.selectbox("Select file for analysis",settings.ENCROACHMENT_DICT.keys())
            file_path = str(settings.ENCROACHMENT_DICT.get(files))
                
            execute_2 = st.button("Begin Analyis")
            if(execute_2):
                try:
                    decrypt_it(file_path, key = auth_token)
                    df = pandas.read_csv(file_path)
                    st.write(df) 
                except:
                    df = pandas.read_csv(file_path)
                    st.write(df)
                if(showGraph):
                    st.subheader("Tracker Id Vs. Location")
                    
                    fig = px.scatter(df, x="Tracker_ID", y="Location", color ="Class")
                    
                    #st.plotly_chart(ff.createdf.plot(kind="scatter", x="Class", y="Location", ax=ax))
                    #plt.savefig("Image_Plot1")
                    #plot_frame = st.empty()
                    #plot_frame.image("Image_Plot1.png")
                    st.plotly_chart(fig)
                    st.subheader("Tracer Id Vs. Class")
                    fig = px.scatter(df, x = "Tracker_ID", y="Class", color = "Location")
                    st.plotly_chart(fig, theme=None)
                    showGraph = False
        
            
        
        if(st.session_state.current == "Visualize"):
            showGraph = True
            
            st.session_state.current = analysis_crit
            st.rerun()
            
        
        
            
            
                    
                #st.write(decrypt_it(file_path, key = auth_token))
    if(st.session_state.current == "Encryption"):
        
        all = list(settings.ENCROACHMENT_DICT.keys())
        all.extend(list(settings.FLOW_DICT.keys()))
        all.extend(list(settings.QUEUE_DICT.keys()))
        files = st.selectbox("Select file for analysis",all)
        if files in settings.ENCROACHMENT_DICT.keys():
            file_path = str(settings.ENCROACHMENT_DICT.get(files))
        elif files in settings.FLOW_DICT.keys():
            file_path = str(settings.FLOW_DICT.get(files))
        else:
            file_path = str(settings.QUEUE_DICT.get(files))
        if(st.button("Encrypt")):    
            encrypt_it(path_csv=file_path)
            st.success("Encryption Successful!")
   
def get_first_frame(video_path, size=CARD_IMAGE_SIZE):
    """Extract the first frame from a video file and resize it to the given size."""
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(size, Image.LANCZOS)
        return pil_image
    else:
        return None

def loadDetections(video_path):
    """
    Ensure the necessary directories exist for the given video path in the detections directory.
    If they don't exist, create them and add an empty sample.txt file.
    """

    video_relative_path = os.path.relpath(video_path, VIDEO_DIR_PATH)
    detections_path = os.path.join('detections', video_relative_path)
    detections_path = os.path.splitext(detections_path)[0] + '.dat'
    detections_dir = os.path.dirname(detections_path)
    print(detections_dir)
    if not os.path.exists(detections_dir):
        os.makedirs(detections_dir, exist_ok=True)
        sample_file_path = os.path.join(detections_dir, 'sample.txt')
        with open(sample_file_path, 'w') as f:
            pass

    if not os.path.exists(detections_path):
        return detections_path, False, []
    else:
        f = open(detections_path,"rb")
        detections = pickle.load(f)
        f.close()
        return detections_path, True, detections

def junctionEvaluation(language):
    global CURRENT_DIR_PATH

    if ("current_dir_path" not in st.session_state):
        st.session_state.current_dir_path = VIDEO_DIR_PATH

    st.text(st.session_state.current_dir_path)
    if ("current_state" not in st.session_state):
        st.session_state.current_state = "homePage"



    if (st.session_state.current_state=="homePage"):
        isVideo = False
        if (st.session_state.current_dir_path.endswith(('.mp4', '.avi','.mov','.AVI'))):
            isVideo = True
            st.session_state.current_state = "checkingDetections"
            st.rerun()
        else:
            st.title("Video Gallery")

        video_files = [f for f in os.listdir(st.session_state.current_dir_path) if f.endswith(('.mp4', '.avi','.mov','.AVI'))]
        folders = [f for f in os.listdir(st.session_state.current_dir_path) if '.' not in f]
        # Display videos in a grid
        cols = st.columns(3)  # Adjust the number of columns as needed
        video_files = folders+video_files
        
        if (st.session_state.current_dir_path!=VIDEO_DIR_PATH):
            if (st.button("Back")):
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind("/")]
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind("/")+1]
                st.rerun()

        for idx, video_file in enumerate(video_files):
            with cols[idx % 3]:  # Change 3 to the number of columns you want
                video_path = os.path.join(st.session_state.current_dir_path, video_file)
                first_frame = get_first_frame(video_path)
                if (idx < len(folders)):
                    first_frame = Image.open(IMAGES_DIR_PATH+"/FolderIcon.png")
                    first_frame = first_frame.resize(CARD_IMAGE_SIZE, Image.LANCZOS)
                    st.image(first_frame, use_column_width=True)
                    st.write(video_file)
                    if st.button(f"Navigate to {video_file}", key=video_file):
                        st.session_state.current_dir_path= st.session_state.current_dir_path+video_file+"/"
                        st.rerun()
                        
                else:
                    if first_frame:
                        st.image(first_frame, use_column_width=True)
                        st.write(video_file)
                        if st.button(f"Analyze {video_file}", key=video_file):
                            st.session_state.current_dir_path = st.session_state.current_dir_path+video_file
                            st.rerun()
                            



    if (st.session_state.current_state=="checkingDetections"):
        col1, col2 = st.columns([3,1])
        with col1:
            st.title("Analysis: " + st.session_state.current_dir_path[st.session_state.current_dir_path.rfind("/")+1:])
        with col2:
            if (st.button("Back to video gallery")):
                st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind('/')+1]
                st.session_state.current_state = "homePage"
                st.rerun()
        detections_path, exists, detections = loadDetections(st.session_state.current_dir_path)
        if (not exists):
            st.subheader("Detections not found!")
            if (st.button("Obtain and save detections")):
                detections = get_detections(st.session_state.current_dir_path)
                saveDetections(detections=detections,filename=detections_path)
                st.rerun()
        else:
            st.session_state.current_state = "DetectionsFound"
            st.rerun()
    
    if (st.session_state.current_state == "DetectionsFound"):
        st.subheader("Detections found!")
        if (st.button("Back to video gallery")):
            st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind('/')+1]
            st.session_state.current_state = "homePage"
            st.rerun()
        col1, col2 = st.columns([1,1])
        with col1:
            if (st.button("Analyze whole junction")):
                st.session_state.current_state = "analyzeWholeJxn" 
                st.rerun()     
        with col2:
            if (st.button("✨ Roadwise Analysisᴮᴱᵀᴬ")): 
                st.session_state.current_state = "roadwiseanalysis"
                st.rerun()

    if (st.session_state.current_state == "analyzeWholeJxn"):
        if (st.button("Back to video gallery")):
            st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind('/')+1]
            st.session_state.current_state = "homePage"
            st.rerun()
        col1,col2 = st.columns([3,1])
        with col1:
            frameThreshold = st.slider("Frame Threshold",min_value=10,max_value=50,value=24)
            if ("frameThreshold" not in st.session_state):
                st.session_state.frameThreshold = frameThreshold
        with col2:
            if (st.button("Detect")):
                st.session_state.current_state = "wholeJunction"
                detections_path, exists, detections = loadDetections(st.session_state.current_dir_path)
                finalDict = cleaningDataset(detections=detections,frameThreshold=st.session_state.frameThreshold)
                st.session_state.finalDict = finalDict
                st.session_state.current_state = "analysis"
                st.rerun()    
        st.subheader("Any vehicle detected in < frameThreshold frames would not be considered in final analysis")

    if (st.session_state.current_state == "analysis"):
        if (st.button("Back to video gallery")):
            st.session_state.current_dir_path = st.session_state.current_dir_path[:st.session_state.current_dir_path.rfind('/')+1]
            st.session_state.current_state = "homePage"
            st.rerun()
        table = calculateWholeJunction(st.session_state.finalDict)
        newTable = {}
        for i in table.keys():
            newTable[model.model.names[i]] = table[i]
            print(model.model.names[i])
        print(newTable)
        df = pandas.DataFrame.from_dict(newTable, orient='index', columns=['Value'])
        st.title('Number of vehicles detected in given clip')
        st.table(df)

    # Specify the directory containing videos

    # Fetch query parameters

    # Fetch all video files


                            


        
    '''# if (len(settings.EVALUATION_DICT.keys()) == 0):
    #     st.sidebar.error("Create a dataset first")
    # else:
    #     source_dir = st.sidebar.selectbox(
    #     "Choose a folder", settings.EVALUATION_DICT.keys())
        
    #     source_path = str(settings.EVALUATION_DICT.get(source_dir))
    #     source_vid = st.sidebar.selectbox(
    #     "Choose a clip", settings.FINAL_DICT[source_dir].keys())
        
        
    #     with open("videos/JunctionEvalDataset/"+source_dir+"/"+source_vid, 'rb') as video_file:
    #         video_bytes = video_file.read()
    #     if video_bytes:
    #         st.video(video_bytes)

    #     threshold = st.sidebar.text_input(
    #         "Enter a integer in range 1-5"
    #     )

    #     try:
            
    #         threshold = int(threshold)
    #         if (threshold > 5 or threshold < 1):
    #             st.sidebar.error("Enter a valid value")
    #         else:
    #             if st.sidebar.button("Start Evaluation"):
    #                 returnVid = "videos/JunctionEvaluations/IndiraNagarClips/clip1.mp4"
    #                 with open(returnVid, 'rb') as video_file2:
    #                     video_bytes2 = video_file2.read()
                        
    #                 if video_bytes2:
    #                     st.video(video_bytes2)
                    
                                                            
    #     except:
    #         st.sidebar.error("Enter a valid integer")'''            

            
                
        
