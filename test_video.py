import cv2
import torch
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification
torch.set_float32_matmul_precision('high')
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification(
    '/home/vinhdeptrai/PycharmProjects/pythonProject/modules/models/pose_classification.pth'
)
video_path = "/home/vinhdeptrai/PycharmProjects/pythonProject/modules/notebooks/4935841_People_Woman_3840x2160.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_width = 640
resize_height = int(height * (resize_width / width))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output4.mp4', fourcc, fps, (resize_width, resize_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (resize_width, resize_height))
    results = detection_keypoint(frame)
    image_draw = frame.copy()
    image_draw = results.plot(boxes=False)
    all_keypoints = results.keypoints.xyn.cpu().numpy()
    for i, (box, keypoint) in enumerate(zip(results.boxes, all_keypoints)):
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        image_draw = cv2.rectangle(
            image_draw,
            (int(x_min), int(y_min)), (int(x_max), int(y_max)),
            (0, 0, 255), 2
        )
        try:
            extracted_keypoint = detection_keypoint.extract_keypoint(keypoint)
            input_classification = torch.tensor(
                extracted_keypoint[10:],
                dtype=torch.float32
            )
            results_classification = classification_keypoint(input_classification)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            label = f'Person {i + 1}: {results_classification.upper()}'
            (w, h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            image_draw = cv2.rectangle(
                image_draw,
                (int(x_min), int(y_min) - 20), (int(x_min) + w, int(y_min)),
                (0, 0, 255), -1
            )
            cv2.putText(
                image_draw,
                label,
                (int(x_min), int(y_min) - 4),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        except Exception as e:
            print(f"Lỗi khi xử lý người thứ {i + 1}: {e}")
    cv2.imshow('Video', image_draw)
    out.write(image_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
