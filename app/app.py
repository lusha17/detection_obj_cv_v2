from typing import List
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pytz
import datetime
import matplotlib.colors as mcolors
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from config import CLASSES_CUSTOM_M, CLASSES_CUSTOM_S, CLASSES_BASE, WEBRTC_CLIENT_SETTINGS
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


#изменим название страницы, отображаемое на вкладке браузера
#set_page_config должна вызываться до всех функций streamlit
st.set_page_config(
    page_title="Weapon Detection Demo",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.sidebar.markdown("""<center data-parsed=""><img src="http://drive.google.com/uc?export=view&id=1Mad62XWdziqcx9wijUODpzGzqYEGhafC" align="center"></center>""",unsafe_allow_html=True,)
st.sidebar.markdown(" ")
st.title('Weapon Detection Demo')


@st.cache(max_entries=3)
def get_yolo5(label):
    if label=='Base':
        return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m.pt')  
    elif label=='Custom small':
        return torch.hub.load('ultralytics/yolov5', 'custom', path='all_s.pt')
    else:
        return torch.hub.load('ultralytics/yolov5', 'custom', path='all_m.pt')

def get_preds(img : np.ndarray) -> np.ndarray:
    return model([img]).xyxy[0].numpy()

def get_colors(indexes : List[int]) -> dict:
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5
    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)
    return color_dict

def get_legend_color(class_name : int):
    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)

class VideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.rgb_colors = rgb_colors
        self.target_class_ids = target_class_ids
        self.confidence_threshold = confidence_threshold

    def get_preds(self, img : np.ndarray) -> np.ndarray:
        return self.model([img]).xyxy[0].numpy()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.get_preds(img)
        result = result[np.isin(result[:,-1], self.target_class_ids)]    
        for bbox_data in result:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            if conf > self.confidence_threshold:
                p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
                img = cv2.rectangle(img, p0, p1, self.rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                class_ = CLASSES[label]
                #if (class_ == 'pistol') | (class_ == 'knife'):
                #    time_detect = datetime.datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None).strftime("%m-%d-%y %H:%M:%S")
                #    self.session_state.append({'object': class_, 'time_detect': time_detect})
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.rgb_colors[label],2,)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


model_type = st.sidebar.selectbox('Select model type',('Base', 'Custom medium', 'Custom small'),index=2)

with st.spinner('Loading the model...'):
    model = get_yolo5(model_type)
st.success('Loading the model.. Done!')

confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)


prediction_mode = st.sidebar.radio("",('Single image', 'Web camera'),index=1)
if model_type == 'Base':
    CLASSES = CLASSES_BASE
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='person')
elif model_type == 'Custom small':
    CLASSES = CLASSES_CUSTOM_S
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='pistol')
else:
    CLASSES = CLASSES_CUSTOM_M
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='pistol')
all_labels_chbox = st.sidebar.checkbox('All classes', value=True)



if all_labels_chbox:
    target_class_ids = list(range(len(CLASSES)))
elif classes_selector:
    target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)
detected_ids = None


if prediction_mode == 'Single image':
    # добавляет форму для загрузки изображения
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'])
    # если файл загружен
    if uploaded_file is not None:
        # конвертация изображения из bytes в np.ndarray
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img)
        #скопируем результаты работы кэшируемой функции, чтобы не изменить кэш
        result_copy = result.copy()
        #отберем только объекты нужных классов
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
        detected_ids = []
        #также скопируем изображение, чтобы не изменить аргумент кэшируемой 
        # функции get_preds
        img_draw = img.copy().astype(np.uint8)
        # нарисуем боксы для всех найденных целевых объектов
        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            if conf > confidence_threshold:
                p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
                img_draw = cv2.rectangle(img_draw, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                class_ = CLASSES[label]
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img_draw = cv2.putText(img_draw, text_for_vis, (int(xtext), int(ytext)),cv2.FONT_HERSHEY_SIMPLEX,0.5,rgb_colors[label],2,)
                detected_ids.append(label)
        # выведем изображение с нарисованными боксами
        # use_column_width растянет изображение по ширине центральной колонки
        st.image(img_draw, use_column_width=True)
elif prediction_mode == 'Web camera':
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False})
    if ctx.video_transformer:
        ctx.video_transformer.model = model
        ctx.video_transformer.rgb_colors = rgb_colors
        ctx.video_transformer.target_class_ids = target_class_ids
        ctx.video_transformer.confidence_threshold = confidence_threshold