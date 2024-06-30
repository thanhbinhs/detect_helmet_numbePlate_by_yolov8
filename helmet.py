import cv2
from ultralytics import YOLO

# Tải model YOLOv8 đã huấn luyện
model = YOLO('./models/helmet.pt')

# Hàm xử lý video
def process_video(video_path, output_path):
    # Mở file video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Lấy các thuộc tính của video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Định nghĩa codec và tạo đối tượng VideoWriter để lưu video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Thực hiện nhận diện mũ bảo hiểm
        results = model(frame)

        # Vẽ các khung bao và nhãn trên khung hình
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0]
            if confidence >= 0.8:
                label = f'Helmet {confidence:.2f}'
                color = (0, 255, 0)  # Green for Helmet
            else:
                label = f'No Helmet {confidence:.2f}'
                color = (0, 0, 255)  # Red for No Helmet

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Ghi khung hình vào video đầu ra
        out.write(frame)

    # Giải phóng các đối tượng video capture và writer
    cap.release()
    out.release()

# Sử dụng ví dụ
input_video_path = './input/video_4.mp4'  # Đường dẫn tới video đầu vào của bạn
output_video_path = './output/out.mp4'  # Đường dẫn để lưu video đầu ra
process_video(input_video_path, output_video_path)
