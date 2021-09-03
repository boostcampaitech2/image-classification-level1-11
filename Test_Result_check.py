#%%
import os
import time
from pathlib import Path
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

start = time.time()

'''
예측 결과가 있는 submission.csv 파일 경로를 아래에서 지정해주고 실행하면 
/opt/mp/Testset Check 디렉토리 내에 test 결과를 5x5 이미지 파일로 저장합니다.
레이블 앞쪽보다는 뒤쪽이 더 예측이 안될거 같아서 뒤에 레이블부터 정렬시키도록 했습니다.

전체 다 저장하는데는 10분 정도 걸립니다. 맨 아래에서 저장할 파일 숫자를 조금 줄이고 테스트해보세요!
한글 깨짐 현상은 링크 달아둔 블로그 참조하시면 됩니다.
'''

dir = Path("/opt/ml/input/data/eval")
submission_file = "/opt/ml/image-classification-level1-11/output/output.csv"
submission = pd.read_csv(submission_file)


label_mapper = {0:"남자, ~29, Mask", 1:"남자, 30~59, Mask", 2:"남자, 60~, Mask",
              3:"여자, ~29, Mask", 4:"여자, 30~59, Mask", 5:"여자, 60~, Mask",
              6:"남자, ~29, incorrect", 7:"남자, 30~59, incorrect", 8:"남자, 60~, incorrect",
              9:"여자, ~29, incorrect", 10:"여자, 30~59, incorrect", 11:"여자, 60~, incorrect",
              12:"남자, ~29, No wear", 13:"남자, 30~59, No wear", 14:"남자, 60~, No wear",
              15:"여자, ~29, No wear", 16:"여자, 30~59, No wear", 17:"여자, 60~, No wear"}


submission["ImageID"] = submission["ImageID"].apply(lambda x: dir/"images"/x)
submission.sort_values(by="ans", ascending=False, inplace=True)

# 한글 깨짐 해결 : https://seonghyuk.tistory.com/31
def save_image_grid_25(n, submission_result, directory):

    predicts = [(Image.open(data), label_mapper[label]) for data, label in submission.values[n*25: (n+1)*25]]

    plt.rcParams["font.family"] = "NanumGothic"
    fig = plt.figure(figsize=(30, 30))
    for i, result in enumerate(predicts):
        ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
        ax.title.set_text(f"{result[1]}")
        ax.title.set_fontsize(32)
        ax.title.set_fontweight("bold")
        plt.imshow(result[0])

    plt.savefig(f"{directory}/sample{n}.jpg")
    plt.close()



os.makedirs("/opt/ml/image-classification-level1-11/Testset Check", exist_ok=True)

# range 안에서 파일 몇개나 저장해줄지 결정.
# i에 따라 25개씩 순서대로 저장.
for i in range(12600//25):

    save_image_grid_25(i, submission, "/opt/ml/image-classification-level1-11/Testset Check")
    if i % 20 == 0:
        print(f"current save {25 * (i+1)} images..")

print(f"all Test Image save complete - {time.time()-start} sec")