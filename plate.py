import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pytesseract
from PIL import Image
import numpy as np
from inference_sdk import InferenceHTTPClient
import tempfile

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

def enhance_image(img):
    # Tăng cường độ sắc nét (Sharpening)
    kernel = np.array([[-1, -1, -1], 
                             [-1, 9, -1], 
                             [-1, -1, -1]])

    sharpened_image = cv2.filter2D(img, -1, kernel)
    imshow("Sharpened Image", sharpened_image)

    return sharpened_image

def ocr_image_from_file(img):
    imshow("Original Image", img)
    
    # # Tăng cường ảnh
    # enhanced_image = enhance_image(img)
    # imshow("Enhanced Image", enhanced_image)

    # Chuyển đổi ảnh sang màu xám
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imshow("Gray Image", gray_image, is_gray=True)

     # Cấu hình Tesseract OCR với các tùy chọn phù hợp
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    detected_text = pytesseract.image_to_string(gray_image, config=custom_config)

    # Hiển thị kết quả OCR
    print("Detected text (raw):", detected_text)

    # Xử lý chuỗi để loại bỏ dấu chấm và gộp các chuỗi lại với nhau
    processed_text = detected_text.replace('.', '').replace('\n', ' ').strip()
    print("Detected text (processed):", processed_text)
    return processed_text

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


def detect_plate(image_path, output_path='./output/out_1.jpeg'):
    # Load YOLO model
    model = YOLO('./models/plate.pt')  # Thay bằng đường dẫn tới mô hình của bạn

    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)

    # Kiểm tra nếu ảnh được tải thành công
    if image is None:
        print("Không thể đọc ảnh từ đường dẫn.")
        return

    # Chuyển đổi ảnh sang RGB vì YOLOv8 mong đợi ảnh RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuyển đổi mảng numpy thành hình ảnh PIL
    pil_img = Image.fromarray(img_rgb)

    # Thực hiện dự đoán trực tiếp
    plate_results = model(pil_img)

    # Xử lý kết quả dự đoán
    for result in plate_results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        confidence = result.conf[0]
        if confidence > 0.2:
            # Cắt vùng ảnh chứa biển số xe
            mini_image = image[y1:y2, x1:x2]
            #save image
            text_ocr = detect_number_plate(mini_image)
            label = text_ocr
            print("Licence: ", label)

            # Vẽ khung chữ nhật và label trên ảnh gốc
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)    

    # Chuyển đổi ảnh sang RGB để hiển thị với matplotlib
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Lưu ảnh vào tệp
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.savefig(output_path)
    print(f"Output saved to {output_path}")

# Sử dụng hàm với đường dẫn đến hình ảnh của bạn
image_path = './input/img_5.png'  # Thay bằng đường dẫn tới hình ảnh của bạn
detect_plate(image_path)
