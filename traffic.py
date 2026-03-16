import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

st.title("🚦 Smart Traffic Pattern Analysis & Signal Timing")

st.write("Upload traffic images to detect vehicles and calculate signal timing.")

# Load YOLO model
model = YOLO("yolov8n.pt")

uploaded_files = st.file_uploader(
    "Upload Traffic Images (Each image = one road)",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

traffic_data = []

if uploaded_files:

    for i,file in enumerate(uploaded_files):

        image = Image.open(file)
        image_np = np.array(image)

        results = model(image_np)

        vehicle_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls)

                label = model.names[cls]

                if label in ["car","bus","truck","motorcycle"]:
                    vehicle_count += 1

        traffic_data.append({
            "Road": f"Road {i+1}",
            "Vehicles": vehicle_count
        })

        st.image(image, caption=f"Road {i+1}")
        st.write(f"Detected Vehicles: {vehicle_count}")

    df = pd.DataFrame(traffic_data)

    st.subheader("Traffic Pattern Data")
    st.dataframe(df)

    total = df["Vehicles"].sum()

    st.subheader("🚦 Recommended Signal Timing")

    for index,row in df.iterrows():

        if total == 0:
            signal_time = 30
        else:
            signal_time = int((row["Vehicles"]/total)*120)

        st.write(f"{row['Road']} → Green Signal: {signal_time} seconds")