import cv2, os, argparse, random, albumentations, torch
import numpy as np
from albumentations.pytorch import ToTensorV2

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    parser.add_argument("--num_segments", type=int, default=9, help="Number of segments")
    parser.add_argument("--frames_per_segment", type=int, default=2, help="Number of frames per segment")
    parser.add_argument("--path_model", type=str, default='best_model.pth', help="Path to the model")
    args = parser.parse_args()
    return args

args = init_parameter()

# Model Initialisation 
import torch
from torch import nn
from torchvision.models import mobilenet_v2

class FireNetAGM(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size = 128, lstm_num_layers = 2, dropout_prob=0.75):
        super(FireNetAGM, self).__init__()

        # Pretrained MobileNetV2 feature extractor
        mobilenet = mobilenet_v2(weights = "DEFAULT")
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])

        # Freeze all the layers of the model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Add Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Batch normalization
        self.bn = nn.BatchNorm1d(1280)

        # LSTM for sequence handling
        self.lstm = nn.LSTM(1280, lstm_hidden_size, lstm_num_layers, batch_first = True)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)

        # Feature extraction
        x = self.backbone(x)
        x = self.gap(x)

        # Apply batch normalization
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = x.view(batch_size, timesteps, -1)

        # Sequence handling
        x, _ = self.lstm(x)

        # Apply dropout
        x = self.dropout(x)

        # Classification for each frame
        x_each_frame = self.classifier(x)
        x_each_frame = torch.sigmoid(x_each_frame)

        # Classification for the sequence (based on the last frame)
        x_last_frame = self.classifier(x[:, -1, :])

        return x_each_frame.squeeze(), x_last_frame.squeeze()

model = FireNetAGM()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(args.path_model))
model.to(device)

# Preprocessing
preprocessing = albumentations.Sequential([
    albumentations.Resize(height=224, width=224, always_apply=True),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225],
                             max_pixel_value=255.,
                             always_apply=True),
    ToTensorV2(),
])

# For all the test videos
for video in os.listdir(args.videos):
    # Set the full path to the video in the format "args.videos/video"
    path_video = os.path.join(args.videos, video)
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    segment_frames = []
    frame_count = 0
    print(video)

    ret, img = cap.read()
    while ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # Convert color space
        tensor_image = preprocessing(image=img)['image']      # Apply transformations
        segment_frames.append(tensor_image.unsqueeze(0))      # Add batch dimension
        frame_count += 1

        # Here you should add your code for applying your method
        if len(segment_frames) == (args.num_segments * args.frames_per_segment):
            with torch.no_grad():
                model.eval()
                X = torch.cat(segment_frames, dim = 0)
                X = X.unsqueeze(0)
                # Predict
                o_each_frame, o_last_frame = model(X.cuda())
                o_last_frame = 1 if torch.sigmoid(o_last_frame) > 0.5 else 0
                if o_last_frame == 1:
                    break
                segment_frames = [] 
        
        ret, img = cap.read()

    cap.release()      

    with open(args.results+video+".txt", "w") as f:
        if o_last_frame == 1:
            # Loop over each sequence in the batch
            fire_start_frames = []
            for i, seq in enumerate(o_each_frame, 1):
                # Find the first frame where the predicted probability of fire is greater than the threshold
                fire_start = 1 if (seq > 0.5) else 0
                if fire_start == 1:
                    t = int((frame_count - i) / fps)
                    f.write(str(t))
                    break