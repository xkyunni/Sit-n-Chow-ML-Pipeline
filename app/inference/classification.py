import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# classify as sit or not sit
class PostureClassifier:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = ["not_sit", "sit"]

        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def predict(self, crop_bgr):
        # return posture label and confidence values
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)

        x = self.transforms(pil).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probs).item())
        pred_label = self.class_names[pred_idx]
        pred_conf = float(probs[pred_idx].item())
        sit_prob = float(probs[self.class_names.index("sit")].item())

        posture = "sitting" if pred_label == "sit" else "not_sitting"

        return {
            "posture": posture,
            "confidence": pred_conf,
            "sit_probability": sit_prob, # for sitting threshold
        }