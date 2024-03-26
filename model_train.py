from imageai.Detection.Custom import DetectionModelTrainer
import torch
torch.cuda.empty_cache()

from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsTinyYOLOv3()
trainer.setDataDirectory(data_directory="/home/dev/Documents/GitHub/S.A.D.A.K/datasets-yolo")
trainer.setTrainConfig(object_names_array=["Sever Cover", "traffic_sign", "Drain Hole", "Pothole"], batch_size=32, num_experiments=100, train_from_pretrained_model="/home/dev/Documents/GitHub/S.A.D.A.K/tiny-yolov3.pt")
trainer.trainModel()