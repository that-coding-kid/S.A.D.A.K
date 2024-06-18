# Python In-built packages
from pathlib import Path
import PIL
# External packages
import streamlit as st
from structures.streamlit_login_auth_ui.widgets import __login__
from structures.essentials import load_model
# Local Modules
import settings
import helper
from locales.settings_languages import COMPONENTS

# Setting page layout
st.set_page_config(
        page_title="S.A.D.A.K",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    ) 

language = st.sidebar.selectbox('Language: ', ["English", "ಕನ್ನಡ", "हिंदी", "বাংলা", "ગુજરાતી","മലയാളം","मराठी","தமிழ்","తెలుగు","اردو","ਪੰਜਾਬੀ","संस्कृत", "অসমীয়া","भोजपुरी","डोगरी","मैथिली","Mizo tawng","Manipuri",])
language_dict = {"English":"en","हिंदी":"hi","ಕನ್ನಡ":"kn","অসমীয়া":"as","বাংলা":"bn","ગુજરાતી":"gu","മലയാളം":"ml","मराठी":"mr","தமிழ்":"ta","తెలుగు":"te","اردو":"ur","ਪੰਜਾਬੀ":"pa","संस्कृत":"sanskrit","भोजपुरी":"bhojpuri","डोगरी":"dogri","मैथिली":"maithili","Mizo tawng":"mizo","Manipuri":"manipuri"}
st.title(COMPONENTS[language_dict[language]]["TITLE"])
__login__obj = __login__(auth_token = "pk_prod_PVY78PYNS84M1SPFKZSCHD1D32BS", 
                    company_name = "S.A.D.A.K",
                    width = 200, height = 250, 
                    logout_button_name = COMPONENTS[language_dict[language]]["LOGOUT"], hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json',
                    language = language_dict[language]
                    )

LOGGED_IN = __login__obj.build_login_ui()


if LOGGED_IN == True:
    # Main page heading
    # Sidebar
    st.sidebar.header(COMPONENTS[language_dict[language]]["CONFIGURATION"])
    helper.startup()
    # Model Options
    model_type = st.sidebar.radio(
        COMPONENTS[language_dict[language]]["MODEL_TYPE"], [COMPONENTS[language_dict[language]]["DETECTION"], COMPONENTS[language_dict[language]]["SEGMENTATION"]])

    confidence = float(st.sidebar.slider(
        COMPONENTS[language_dict[language]]["CONFIDENCE"], 25, 100, 40)) / 100

    # Selecting Detection Or Segmentation
    if model_type == COMPONENTS[language_dict[language]]["DETECTION"]:
        model_path = Path(settings.DETECTION_MODEL)
    elif model_type == COMPONENTS[language_dict[language]]["SEGMENTATION"]:
        model_path = Path(settings.SEGMENTATION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = load_model(model_path)
    except Exception as ex:
        st.error(COMPONENTS[language_dict[language]]["LOAD_ERROR"]+model_path)
        st.error(ex)

    st.sidebar.header(COMPONENTS[language_dict[language]]["CONFIG_SUBTITLE"])
    source_radio = st.sidebar.radio(
        COMPONENTS[language_dict[language]]["SELECT_SOURCE"], [COMPONENTS[language_dict[language]]["IMAGE"],COMPONENTS[language_dict[language]]["VIDEO"],COMPONENTS[language_dict[language]]["RTSP"],COMPONENTS[language_dict[language]]["YOUTUBE"],COMPONENTS[language_dict[language]]["ENCROACHMENT"],COMPONENTS[language_dict[language]]["JUNCTION"],COMPONENTS[language_dict[language]]["JUNCTIONEVAL"],COMPONENTS[language_dict[language]]["BENCHMARKING"],"Analyze"])

    source_img = None
    # If image is selected
    if source_radio == COMPONENTS[language_dict[language]]["IMAGE"]:
        source_img = st.sidebar.file_uploader(
            COMPONENTS[language_dict[language]]["SOURCE_IMG"], type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error(COMPONENTS[language_dict[language]]["IMG_ERROR"])
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_column_width=True)
            else:
                if st.sidebar.button(COMPONENTS[language_dict[language]]["DETECT_OBJ"]):
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    try:
                        with st.expander(COMPONENTS[language_dict[language]]["DETECTION_RES"]):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        # st.write(ex)
                        st.write(COMPONENTS[language_dict[language]]["NO_IMG"])

    elif source_radio == COMPONENTS[language_dict[language]]["VIDEO"]:
        helper.play_stored_video(confidence, model,language_dict[language])

    elif source_radio == COMPONENTS[language_dict[language]]["RTSP"]:
        helper.play_rtsp_stream(confidence, model,language_dict[language])

    elif source_radio == COMPONENTS[language_dict[language]]["YOUTUBE"]:
        helper.play_youtube_video(confidence, model,language_dict[language])
        
    elif source_radio == COMPONENTS[language_dict[language]]["ENCROACHMENT"]:
        helper.enchroachment(confidence,language_dict[language])
        
    elif source_radio == COMPONENTS[language_dict[language]]["JUNCTION"]:  
        helper.junctionEvaluationDataset(language_dict[language])
        
    elif source_radio == COMPONENTS[language_dict[language]]["JUNCTIONEVAL"]:
        helper.junctionEvaluation(language_dict[language])
        
    elif source_radio == COMPONENTS[language_dict[language]]["BENCHMARKING"]:
        helper.benchMarking(confidence,language_dict[language])
    elif source_radio == "Analyze":
        helper.Analyze(language_dict[language])
    else:
        st.error("Please select a valid source type!")
