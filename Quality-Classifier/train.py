import random 
import torch
import sys

sys.path.append('../')
from Helper_Modules import model_setups, engine
# set the seeds
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Device agnostic code 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

run_number = 0

print(f"\n---------------------------------------------run:{run_number}---------------------------------------------\n")
torch.manual_seed(42)
model_ = model_setups.efficientNet_b0(False, 2, device)
training_dict = dict(project_name="Quality-Classifier-1",
                     artifact_name="tr_v0_32_1e5_5e5",
                     artifact_discription="hyperparameter experiments",
                     dataset="quality_0:v0",
                     download_data=False,
                     batch_size=32,
                     run_number=run_number,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,   # just to download it and keep it in artifacts locally
                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
                                                 {'params': model_.classifier.parameters(), 'lr': 0.00005}]
                                                 , lr=0.00001),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=30,
                     class_names=["bad","good"],
                     device=device)

trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"\n---------------------------------------------run:{run_number}---------------------------------------------\n")
torch.manual_seed(42)
model_ = model_setups.efficientNet_b0(False, 2, device)
training_dict = dict(project_name="Quality-Classifier-1",
                     artifact_name="tr_v0_32_5e5",
                     artifact_discription="hyperparameter experiments",
                     dataset="quality_0:v0",
                     download_data=False,
                     batch_size=32,
                     run_number=run_number,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,   # just to download it and keep it in artifacts locally
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=30,
                     class_names=["bad","good"],
                     device=device)

trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"\n---------------------------------------------run:{run_number}---------------------------------------------\n")
torch.manual_seed(42)
model_ = model_setups.efficientNet_b0(False, 2, device)
training_dict = dict(project_name="Quality-Classifier-1",
                     artifact_name="tr_v0_32_1e5",
                     artifact_discription="hyperparameter experiments",
                     dataset="quality_0:v0",
                     download_data=False,
                     batch_size=32,
                     run_number=run_number,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,   # just to download it and keep it in artifacts locally
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=30,
                     class_names=["bad","good"],
                     device=device)

trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"\n---------------------------------------------run:{run_number}---------------------------------------------\n")
torch.manual_seed(42)
model_ = model_setups.efficientNet_b0(False, 2, device)
training_dict = dict(project_name="Quality-Classifier-1",
                     artifact_name="tr_v0_32_5e6_1e5",
                     artifact_discription="hyperparameter experiments",
                     dataset="quality_0:v0",
                     download_data=False,
                     batch_size=32,
                     run_number=run_number,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,   # just to download it and keep it in artifacts locally
                     optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
                            {'params': model_.classifier.parameters(), 'lr': 0.00001}]
                            , lr=0.000005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=30,
                     class_names=["bad","good"],
                     device=device)

trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"\n---------------------------------------------run:{run_number}---------------------------------------------\n")
torch.manual_seed(42)
model_ = model_setups.efficientNet_b0(False, 2, device)
training_dict = dict(project_name="Quality-Classifier-1",
                     artifact_name="tr_v0_32_1e5",
                     artifact_discription="hyperparameter experiments",
                     dataset="quality_0:v0",
                     download_data=False,
                     batch_size=32,
                     run_number=run_number,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,   # just to download it and keep it in artifacts locally
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["bad","good"],
                     device=device)

trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"\n---------------------------------------------run:{run_number}---------------------------------------------\n")
torch.manual_seed(42)
model_ = model_setups.efficientNet_b0(False, 2, device)
training_dict = dict(project_name="Quality-Classifier-1",
                     artifact_name="tr_v0_64_1e5",
                     artifact_discription="continue trainng",
                     dataset="quality_0:v0",
                     download_data=False,
                     batch_size=64,
                     run_number=run_number,
                     model_name="tr_v0_32_1e5:v1",
                     model_filename="trained_model.pth",
                     continue_training=True,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["bad","good"],
                     device=device)

trained_model = engine.train_and_log(model_, training_dict)




