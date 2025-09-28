# app.py
import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
import io

st.title("تطبيق تأثير Reinhard على الصور")

# رفع الصور
source_file = st.file_uploader("اختر صورة التأثير (source)", type=["jpg", "png", "jpeg"])
target_file = st.file_uploader("اختر الصورة الأصلية (target)", type=["jpg", "png", "jpeg"])

def get_mean_std(x):
    x_mean, x_std = cv.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std  = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

def reinhard_transfer(target_img, source_mean, source_std):
    target_mean, target_std = get_mean_std(target_img)
    normalized = (target_img - target_mean) / target_std
    transferred = (normalized * source_std) + source_mean
    transferred = np.clip(transferred, 0, 255)
    return transferred.astype(np.uint8)

if st.button("تطبيق التأثير"):
    if source_file and target_file:
        # قراءة صورة التأثير
        source_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
        source_img = cv.imdecode(source_bytes, cv.IMREAD_COLOR)
        source_img = cv.cvtColor(source_img, cv.COLOR_BGR2LAB)
        source_mean, source_std = get_mean_std(source_img)

        # قراءة الصورة الأصلية
        target_bytes = np.asarray(bytearray(target_file.read()), dtype=np.uint8)
        target_img = cv.imdecode(target_bytes, cv.IMREAD_COLOR)
        target_img = cv.cvtColor(target_img, cv.COLOR_BGR2LAB)

        # تطبيق التأثير
        modified_img = reinhard_transfer(target_img, source_mean, source_std)
        modified_img = cv.cvtColor(modified_img, cv.COLOR_LAB2BGR)

        # عرض الصورة الناتجة
        st.image(modified_img, channels="BGR", caption="الصورة بعد التأثير")
    else:
        st.warning("يرجى رفع كلتا الصورتين أولاً")
