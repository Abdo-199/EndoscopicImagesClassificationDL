import random 
import torch
import sys

sys.path.append('../')
from Helper_Modules import model_setups, engine

# set the seeds
torch.backends.cudnn.deterministic = True
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Device agnostic code 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# print("\n---------------------------------------------run:0---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_1e4",
#                      artifact_discription="one learning rate for all the layers",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.0001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:1---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_1e5",
#                      artifact_discription="less learning rate to keep the models features",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:2---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_1e6",
#                      artifact_discription="less learning rate",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.000001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:3---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_1e5_1e6",
#                      artifact_discription="higher learning rate for the classifier, and less for all other layers",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00001}]
#                                                  , lr=0.000001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:4---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_5e5_5e6",
#                      artifact_discription="encreasing both learning rates",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:5---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_5e5_5e6",
#                      artifact_discription="encreasing the batch size to 64",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:6---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_5e5_5e6",
#                      artifact_discription="encreasing the batch size to 64",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:7---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_5e5_5e6",
#                      artifact_discription="encreasing the batch size to 64",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_1e5",
#                      artifact_discription="encreasing the batch size to 64",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_1e4",
#                      artifact_discription="encreasing the batch size to 64",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.0001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_1e5_5e5",
#                      artifact_discription="specific lr 5e5 classifier, 1e5 all batch size 32",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_5e5",
#                      artifact_discription="learning rate 5e5 to all",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_25e6",
#                      artifact_discription="learning rate 5e5 to all",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.000025),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_1e5_continue",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="tr_v0_32_1e5_5e5:v1",
#                      model_filename="trained_model.pth",
#                      continue_training=True,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_32_1e5_5e5",
#                      artifact_discription="specific lr 5e5 classifier, 1e5 all batch size 32",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_1e5_continue",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="tr_v0_32_1e5_5e5:v2",
#                      model_filename="trained_model.pth",
#                      continue_training=True,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_5e6_continue",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="tr_v0_32_1e5_5e5:v2",
#                      model_filename="trained_model.pth",
#                      continue_training=True,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v0_64_1e5_5e6_continue",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_0:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="tr_v0_32_1e5_5e5:v2",
#                      model_filename="trained_model.pth",
#                      continue_training=True,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00001}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v1_32_1e5_5e5",
#                      artifact_discription="specific lr 5e5 classifier, 1e5 all batch size 32",
#                      dataset="quality_1:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v1_32_1e5_5e5",
#                      artifact_discription="specific lr 5e5 classifier, 1e5 all batch size 32",
#                      dataset="quality_1:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v1_64_1e5",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_1:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=True,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)



# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v1_64_1e5_continue",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_1:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="tr_v1_32_1e5_5e5:v0",
#                      model_filename="trained_model.pth",
#                      continue_training=True,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=15,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v2_32_1e5_5e5",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_2:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:8---------------------------------------------\n")
# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict = dict(project_name="Quality-classifier",
#                      artifact_name="tr_v2_64_1e5_5e5_sqrt2",
#                      artifact_discription="continue training the best performing 32 batch model, with a batch size of 64 and smaller lr",
#                      dataset="quality_2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00007}]
#                                                  , lr=0.000015),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model = engine.train_and_log(model_, training_dict)

# print("\n---------------------------------------------run:0---------------------------------------------\n")
# torch.manual_seed(42)
# model_0 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_0 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_raw",
#                      artifact_discription="traind on a non transformed dataset",
#                      dataset="raw_quality:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model=model_0,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,   # just to download it and keep it in artifacts locally
#                      optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.0001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=20,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_0 = engine.train_and_log(model_0, training_dict_0)

# print("\n---------------------------------------------run:1---------------------------------------------\n")

# torch.manual_seed(42)
# model_1 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trans",
#                      artifact_discription="traind on a transformed dataset as recommended for efficientnet",
#                      dataset="auto_transforms_efficientNet:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model=model_1,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_1.parameters(),lr=0.0001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=10,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_1, training_dict_trans_1)

# print("\n---------------------------------------------run:2---------------------------------------------\n")

# torch.manual_seed(42)
# model_1 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trans",
#                      artifact_discription="traind on a transformed dataset as recommended for efficientnet",
#                      dataset="auto_transforms_efficientNet:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model=model_1,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_1.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=20,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_1, training_dict_trans_1)

# print("\n---------------------------------------------run:3---------------------------------------------\n")

# torch.manual_seed(42)
# model_1 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_20",
#                      artifact_discription="20 epochs, 0.0001 lr, ",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model=model_1,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_1.parameters(),lr=0.0001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=20,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_1, training_dict_trans_1)

# print("\n---------------------------------------------run:4---------------------------------------------\n")

# torch.manual_seed(42)
# model_1 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_50",
#                      artifact_discription="50 epochs, 0.00005 lr, ",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model=model_1,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_1.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_1, training_dict_trans_1)

# print("\n---------------------------------------------run:5---------------------------------------------\n")

# torch.manual_seed(42)
# model_1 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_50",
#                      artifact_discription="50 epochs, 0.00005 lr, batchsize 64",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model=model_1,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_1.parameters(),lr=0.000001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_1, training_dict_trans_1)

# print("\n---------------------------------------------run:6---------------------------------------------\n")

# torch.manual_seed(42)
# model_1 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_50",
#                      artifact_discription="50 epochs, 0.00005 lr, batchsize 64",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model=model_1,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_1.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_1, training_dict_trans_1)

# print("\n---------------------------------------------run:7---------------------------------------------\n")

# torch.manual_seed(42)
# model_1 = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_50",
#                      artifact_discription="50 epochs, 0.00005 lr, batchsize 128",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model=model_1,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam(params=model_1.parameters(),lr=0.00005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_1, training_dict_trans_1)

# print("\n---------------------------------------------run:7---------------------------------------------\n")

# torch.manual_seed(42)
# model_frozen = model_setups.efficientNet_b0(True, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_frozen",
#                      artifact_discription="Continue training with freezing all the layers but classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="EfNtB0_tr_trs_50:v4",
#                      model_filename="trained_model.pth",
#                      continue_training=True,
#                      optimizer=torch.optim.Adam(params=model_frozen.parameters(),lr=0.0001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=10,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_frozen, training_dict_trans_1)


# print("\n---------------------------------------------run:8---------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using higher lr for the classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:9---------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using higher lr for the classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:10--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using higher lr for the classifier and encreasing the lr of features",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:11--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using higher lr for the classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:11--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using higher lr for the classifier and encreasing the lr of features",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.001}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:13--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using lower lr for the classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=100,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:14--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using lower lr for the classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:15--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using lower lr for the classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:16--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="using lower lr for the classifier",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)


# print("\n---------------------------------------------run:17--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="repeating 54 without transfer",
#                      dataset="auto_trans_effNet_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)
# print("\n---------------------------------------------run:18--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="first run with the stretched data",
#                      dataset="trans_quality_v4:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.0001}]
#                                                  , lr=0.00001),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:19--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="first run with the stretched data",
#                      dataset="trans_quality_v4:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=30,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:20--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="first run with the stretched data",
#                      dataset="trans_quality_v4:v0",
#                      download_data=False,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:21--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="EfNtB0_tr_trs_perLay",
#                      artifact_discription="first run with the stretched data",
#                      dataset="trans_quality_v4:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:22--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="trans_v2_64_5e6_5e5",
#                      artifact_discription="test data not stretched",
#                      dataset="trans_quality_v2:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:23--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="trans_v3_64_5e6_5e5",
#                      artifact_discription="test data stretched",
#                      dataset="trans_quality_v3:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:24--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="trans_v4_64_5e6_5e5",
#                      artifact_discription="test data not stretched",
#                      dataset="trans_quality_v4:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:25--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="trans_v4_32_5e6_5e5",
#                      artifact_discription="test data stretched",
#                      dataset="trans_quality_v4:v0",
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
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

# print("\n---------------------------------------------run:24--------------------------------------------\n")

# torch.manual_seed(42)
# model_ = model_setups.efficientNet_b0(False, 2, device)
# training_dict_trans_1 = dict(project_name="Quality-classifier",
#                      artifact_name="trans_v4_64_5e6_5e5",
#                      artifact_discription="test data not stretched",
#                      dataset="trans_quality_v4:v0",
#                      download_data=False,
#                      batch_size=64,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=60,
#                      class_names=["bad","good"],
#                      device=device)

# trained_model_trans_1 = engine.train_and_log(model_, training_dict_trans_1)

