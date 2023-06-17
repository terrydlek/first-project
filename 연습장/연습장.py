import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

# 모델 로드
weights = '/Users/USER/Desktop/rescue/yolov5/runs/train/yolov5x_results/weights/best.pt'  # 모델의 가중치 파일 경로
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())

# 입력 이미지 크기 및 임계값 설정
img_size = 640  # 입력 이미지 크기
conf_thres = 0.3  # 객체 탐지 임계값
iou_thres = 0.5  # 경계 상자 IoU 임계값

# 클래스 레이블
class_labels = ['Person drowning']  # 탐지할 클래스 레이블

# 웹캠 또는 비디오 파일 열기
video_path = 0  # 웹캠 사용 시: 0, 비디오 파일 사용 시: 'video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 사이즈 조정 및 전처리
    img = cv2.resize(frame, (img_size, img_size))
    img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # 객체 탐지 수행
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # 탐지된 객체 확인
    for det in pred:
        if det is not None and len(det):
            # 클래스와 경계 상자 정보 추출
            classes = det[:, -1].long()
            for i, class_idx in enumerate(classes):
                class_label = class_labels[class_idx]
                if class_label == 'Person drowning':
                    print(1)  # 'Person drowning' 객체가 탐지되면 1 출력

    # 화면에 이미지 및 결과 표시
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()