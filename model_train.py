from imageai.Detection.Custom import DetectionModelTrainer
import torch
torch.cuda.empty_cache()

CUDA_LAUNCH_BLOCKING=1
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="datasets-yolo")
trainer.setTrainConfig(object_names_array=["potholes","traffic_sign","unknown"], batch_size=32, num_experiments=100, train_from_pretrained_model="yolov3.pt")
trainer.trainModel()