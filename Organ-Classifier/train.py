import sys

import torch

sys.path.append("../")
from Helper_Modules import model_setups, engine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
run_number = 0 

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device, False)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v0_32_scratch",
                     artifact_discription="learning from scatch",
                     dataset="organ_0:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v0_32_transfer",
                     artifact_discription="transfer learning",
                     dataset="organ_0:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v0_32_5e6",
                     artifact_discription="suitable parameter",
                     dataset="organ_0:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.000005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v0_64_5e5",
                     artifact_discription="suitable parameters",
                     dataset="organ_0:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=64,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v0_64_1e5",
                     artifact_discription="suitable parameter",
                     dataset="organ_0:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=64,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00001),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v0_32_5e5",
                     artifact_discription="suitable parameter",
                     dataset="organ_0:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=15,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v0_64_5e5",
                     artifact_discription="uncropped",
                     dataset="organ_0:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=64,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=20,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v1_64_5e5",
                     artifact_discription="train cropeed",
                     dataset="organ_1:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=64,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=20,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v2_64_5e5",
                     artifact_discription="all cropeed",
                     dataset="organ_2:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=64,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False, 
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=20,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}----------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v3_32_5e5",
                     artifact_discription="suitable parameter",
                     dataset="organ_3:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam(params=model_.parameters(),lr=0.00005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=50,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v3_32_5e6_5e5",
                     artifact_discription="suitable parameter",
                     dataset="organ_3:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
                                                 {'params': model_.classifier.parameters(), 'lr': 0.00005}]
                                                 , lr=0.000005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=50,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v3_64_5e6_5e5",
                     artifact_discription="suitable parameter",
                     dataset="organ_3:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=64,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
                                                 {'params': model_.classifier.parameters(), 'lr': 0.00005}]
                                                 , lr=0.000005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=50,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v3_32_5e6_5e5",
                     artifact_discription="imbalanced dataset",
                     dataset="organ_3:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
                                                 {'params': model_.classifier.parameters(), 'lr': 0.00005}]
                                                 , lr=0.000005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=100,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v4_32_5e6_5e5",
                     artifact_discription="balanced dataset",
                     dataset="organ_4:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
                                                 {'params': model_.classifier.parameters(), 'lr': 0.00005}]
                                                 , lr=0.000005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=100,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

run_number+=1

print(f"---------------------------------------------run:{run_number}---------------------------------------------")
model_ = model_setups.efficientNet_b0(False, 3, device)
training_dict = dict(project_name="Organ-Classifier-1",
                     artifact_name="tr_v4_32_5e6_5e5",
                     artifact_discription="balanced dataset",
                     dataset="organ_4:v0",
                     download_data=False,
                     run_number=run_number,
                     batch_size=32,
                     model_name="efficientNet_b0:v0",
                     model_filename="initialized_model.pth",
                     continue_training=False,
                     optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
                                                 {'params': model_.classifier.parameters(), 'lr': 0.00005}]
                                                 , lr=0.000005),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     epochs=50,
                     class_names=["Ear","Nose","VocalFolds"],
                     device=device)
trained_model = engine.train_and_log(model_, training_dict)

# run_number+=1

# print(f"---------------------------------------------run:{run_number}---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier-1",
#                      artifact_name="tr_v5_32_5e6_5e5",
#                      artifact_discription="balanced dataset",
#                      dataset="organ_5:v0",
#                      download_data=False,
#                      run_number=run_number,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# run_number+=1

# print(f"---------------------------------------------run:{run_number}---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier-1",
#                      artifact_name="tr_v6_32_5e6_5e5",
#                      artifact_discription="balanced dataset",
#                      dataset="organ_6:v0",
#                      download_data=False,
#                      run_number=run_number,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)

# run_number+=1

# print(f"---------------------------------------------run:{run_number}---------------------------------------------")
# model_ = model_setups.efficientNet_b0(False, 3, device)
# training_dict = dict(project_name="Organ-Classifier-1",
#                      artifact_name="tr_v7_32_5e6_5e5",
#                      artifact_discription="balanced dataset",
#                      dataset="organ_7:v0",
#                      download_data=False,
#                      run_number=run_number,
#                      batch_size=32,
#                      model_name="efficientNet_b0:v0",
#                      model_filename="initialized_model.pth",
#                      continue_training=False,
#                      optimizer=torch.optim.Adam([{'params': model_.features.parameters()},
#                                                  {'params': model_.classifier.parameters(), 'lr': 0.00005}]
#                                                  , lr=0.000005),
#                      loss_fn=torch.nn.CrossEntropyLoss(),
#                      epochs=50,
#                      class_names=["Ear","Nose","VocalFolds"],
#                      device=device)
# trained_model = engine.train_and_log(model_, training_dict)