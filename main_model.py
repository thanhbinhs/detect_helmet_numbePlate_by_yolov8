import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import logging
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt


# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="0wYbnSoNAgV8o4VP6c2L"
)

def imshow(title = "", image = None, is_gray = False):
    if(is_gray):
        plt.imshow(image, cmap='gray')
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.title(title)
    plt.show()

def detect_number_plate(img):
    # infer on a local image
    result = CLIENT.infer(img, model_id="image_numberplate_1/2")
    text = ''

    # Parse the result to get bounding boxes
    predictions = result['predictions']

    # Draw bounding boxes on the image
    for prediction in predictions:        
        # Add label to the bounding box
        label = prediction['class']
        text += label
    
    return text

def detect_objects_video(plate_model, helmet_model, video_path, output_path='./output/video_model.mp4', device='cuda'):
    plate_model.to(device)
    helmet_model.to(device)

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video at path '{video_path}' does not exist or cannot be opened.")
    
    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Đặt codec và tạo VideoWriter để lưu video kết quả
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Tạm thời thay đổi cấu hình ghi log
    logging.getLogger().setLevel(logging.ERROR)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chuyển đổi ảnh sang RGB vì YOLOv8 mong đợi ảnh RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Chuyển đổi mảng numpy thành hình ảnh PIL
        pil_img = Image.fromarray(img_rgb)
  

        # Thực hiện dự đoán trực tiếp 
        plate_results = plate_model(pil_img)
        # Thực hiện nhận diện mũ bảo hiểm
        helmet_result = helmet_model(img_rgb)

        # In thông tin dự đoán
        print(f"image {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

                # Vẽ các khung bao và nhãn trên khung hình
        for result in helmet_result[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0]
            #
            if confidence >= 0.6:
                label = f'Helmet {confidence:.2f}'
                color = (0, 255, 0)  # Green for Helmet
            else:
                label = f'No Helmet {confidence:.2f}'
                color = (0, 0, 255)  # Red for No Helmet

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

         # Xử lý kết quả dự đoán
        for result in plate_results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0]
            # if confidence > 0.2:
            # Cắt vùng ảnh chứa biển số xe
            mini_image = img_rgb[y1:y2, x1:x2]
            #save image
            text_ocr = detect_number_plate(mini_image)
            label = text_ocr
            print("Licence: ", label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



        # Ghi khung hình vào video đầu ra
        out.write(frame)

        # Giải phóng bộ nhớ GPU không cần thiết
        torch.cuda.empty_cache()

    # Khôi phục cấu hình ghi log
    logging.getLogger().setLevel(logging.INFO)

    # Giải phóng các tài nguyên
    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")

# Khởi tạo mô hình một lần duy nhất
plate_model = YOLO('./models/plate.pt')
helmet_model = YOLO('./models/helmet.pt')
plate_model.model.fuse()  # Fuse layers only once
helmet_model.model.fuse()  # Fuse layers only once

# Sử dụng hàm với đường dẫn đến video của bạn
video_path = './input/video_4.mp4'  # Thay bằng đường dẫn tới video của bạn

# Chạy trên CPU
detect_objects_video(plate_model, helmet_model, video_path, device='cuda')
