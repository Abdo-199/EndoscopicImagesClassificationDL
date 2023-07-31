# Deep Learning-Based Image Classification for Endoscopic Images
## How to use
- Create a conda envronment with the requiered packages to run the code
```bash
conda create --name endo_v1 python=3.10.11
conda activate endo_v1
pip install -r requirements.txt
```
### wandb
Weights and Biases is used in this project to organize the data and the models and to track the experiments.
Each Classifier has a project in WandB. The projects are open for public to view, but to submit runs or you have to be invited to the team.
    - [quality-classifier](https://wandb.ai/d-ml/quality-classifier-1/overview?workspace=user-elsharkawi99)
    - [organ-classifier](https://wandb.ai/d-ml/Organ-Classifier-1/overview?workspace=user-elsharkawi99)
When you log WandB runs for the first time from your PC, you will be asked to inter your API Key which can be found under user settings in WandB
### Create datasets 
- The central dataset can be downloaded from [allData](https://drive.google.com/drive/folders/17NOXGLKOTXT_f0_xuNggdmOFUcOs4GrM?usp=sharing).
- After downloading allData place it in 2023 Endoscopy Classification Elsharkawi/data.
- By running the code in the notebooks `Organ-Classifier/create_artifacts` and `Quality-Classifier/create_artifacts` all the datasets of the Organ-Classifier and Quality-Classifer will be created under `Organ-Classifeir/data` and `Quality-Classifeir/data`, and logged into WandB.
- At the end of each create_artifacts.ipynb the base model EfficientnetB0 will be logged to the WandB project and the training code is ready to be executed
### Training
- Under each classifier there is a train.py script with all the experiments shown in the thesis.
- If you have write access to the WandB projects, the scripts will be runable without recreating the datasets, as the datasets are already logged into the WandB projects. 
## Flask App
to run the flask app in debug mode:
```bash
cd Flask-App/server
flask --app main.py --debug run
```
to deploy the app to a docker image:
First change `trained_organ.pth` and `trained_quality.pth` to `server/trained_organ.pth` and `server/trained_quality.pth`
```bash
cd Flask-App
docker build -t endoscopy_classifier . && docker run --rm -it -p 5000:5000 endoscopy_classifier
```
## Usage of framesDisassembler.py
```bash
python framesDisassembler.py Video_path output_path Excel_path device quality organ frame-rate
```