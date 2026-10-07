import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image


# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)

print("Loading weights...")

try:
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print("Model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model = model.to(device)
model.eval()


# Preprocess Transform (300x300 + ImageNet normalization, as in training)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load Face Detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open Video Source
# Option 1: video file (default)
cap = cv2.VideoCapture("C:\\Users\\F\\PycharmProjects\\smile-detection\\Can_You_Spot_A_Fake_Smile_7c21660a_9369_4424_a716_e3d02275c5f2.mp4")

# Option 2: phone camera over Wi-Fi (e.g. IP Webcam app); uncomment and set your phone's address
# cap = cv2.VideoCapture("http://<phone-ip>:8080/video")

# Option 3: laptop webcam;
# cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print(" ERROR: Cannot open video file.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break  # video finished

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            _, pred = torch.max(output, 1)
            label = "Smile" if pred.item() == 1 else "No Smile"

        # draw box + label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if label == "Smile" else (0, 0, 255), 2)

    cv2.imshow("Smile Detection (Video Input)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
