import random 
import sys

import torch

sys.path.append("../")
from Helper_Modules import model_setups, engine

# Device agnostic code 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# print("---------------------------------------------run:0---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device, False)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v0_32_scratch",
#                      artifact_discription="learning from scatch",
#                      dataset="organ_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)


# print("---------------------------------------------run:1---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v0_32_transfer",
#                      artifact_discription="transfer learning, ",
#                      dataset="organ_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:2---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v1_32_traincrop",
#                      artifact_discription="Black frame is removed from train data",
#                      dataset="organ_1:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:3---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v2_32_allcrop",
#                      artifact_discription="all the frames are cropped",
#                      dataset="organ_2:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:4---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device, False)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v0_32_scratch",
#                      artifact_discription="all the frames are cropped, learning from scratch",
#                      dataset="organ_2:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:5----------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v3_32_imbalanced",
#                      artifact_discription="train on the new dataset, but imbalanced, and noisy",
#                      dataset="organ_3:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:6----------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v4_32_balanced",
#                      artifact_discription="train on the new dataset, balanced but noisy",
#                      dataset="organ_4:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:7----------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v5_32_clean",
#                      artifact_discription="train on the new dataset, balanced and clean",
#                      dataset="organ_5:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)


# print("---------------------------------------------run:5----------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v3_32_imbalanced",
#                      artifact_discription="train on the new dataset, but imbalanced, and noisy",
#                      dataset="organ_3:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:6----------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v4_32_balanced",
#                      artifact_discription="train on the new dataset, balanced but noisy",
#                      dataset="organ_4:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:7----------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v5_32_clean",
#                      artifact_discription="train on the new dataset, balanced and clean",
#                      dataset="organ_5:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# print("---------------------------------------------run:0---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device, False)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v0_32_scratch",
#                      artifact_discription="learning from scatch",
#                      dataset="organ_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)


# print("---------------------------------------------run:1---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier",
#                      artifact_name="tr_v0_32_transfer",
#                      artifact_discription="transfer learning",
#                      dataset="organ_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)