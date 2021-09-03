## 팀원

김지성 오동규 윤채원 이한빈 장동주 전미원 최윤성

# 프로젝트 개요

- COVID-19의 확산을 막기 위해 딥러닝 모델을 이용해 마스크 착용 여부를 판단하려고 합니다. 공공장소 입구에 놓인 카메라를 이용해 마스크를 제대로 썼는지 확인하고, 추가적으로 성별 및 연령대에 대한 구분도 진행합니다.
- Class Description

![56bd7d05-4eb8-4e3e-884d-18bd74dc4864..png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/807d6c1c-7d16-4d05-b8ea-067f0c75ee5c/56bd7d05-4eb8-4e3e-884d-18bd74dc4864..png)

# Set up

### Install Requirements

- `pip install -r requirements.txt`

### dependencies

```markdown
torch>=1.7.1
torchvision>=0.8.2
albumentations>=1.0.3
matplotlib>=3.2.1
opencv-python>=4.5.1.48
pandas>=1.1.5
scikit-learn>=0.24.2
seaborn>=0.11.2
tensorboard>=2.4.1
timm>=0.4.12
```

# 디렉토리 구조

```markup
train/
	|___ train.py  :  단일 모델을 학습시킵니다.
  |___ train_with_cutmix_and_sam.py  :  학습 데이터에 대해 cutmix를 적용할 수 있고,
	|                                     sam optimzier를 이용한 학습이 가능합니다.
  |___ train_multi_model.py : Gender/Mask/Age 각각의 클래스를 예측하는 모델 학습을 위한 script

inference/
  |___ inference.py : 학습이 완료된 모델의 Test set에 대한 예측 결과 저장.
  |___ inference_multi_model.py : Gender/Mask/Age 각각의 클래스를 예측한 결과를 원래의 레이블로 맵핑.

model.py 

dataset.py : 기본적인 dataset fetch/load proedure 정의, Data Augmentation Method 구현.
AAFMask.py : AAF Dataset 사용 script

loss.py : Focal Loss, LabelSmoothingLoss, F1 Loss 적용 with smooth parameter

opt.py : torch.optim module + sam optimizer 추가

util.py 

config.json / config_multi_model.json : model train에 필요한 hyper parameter setting

Test_Result_check.py : Test set inference 결과를 레이블과 함께 이미지 파일로 저장하는 script
```

# Train

- train single model

    `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python ./train/train.py`

- train Multi model

    `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python ./train/train_multi_model.py`

- train single model with CutMix, SAM optimizer

    `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python ./train/train with cutmix and sam.py`

# inference

- inference single model

    `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python ./inference/inference.py`

- inference multi model

    `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python ./inference/inference_multi_model.py`
