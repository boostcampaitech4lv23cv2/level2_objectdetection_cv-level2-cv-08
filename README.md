# 🌏 프로젝트 개요

<img width="809" alt="Untitled" src="https://user-images.githubusercontent.com/74412423/206112344-eff89532-32cc-49f1-b5ad-dece36f22aa0.png">


대량 생산, 대량 소비의 시대에서 ‘쓰레기 대란’, ‘매립지 부족’과 같은 사회 문제가 발생하고 있다. 이러한 문제를 해결하기 위해서 분리수거는 환경 부담을 줄일 수 있는 방법이다. 잘 분리수거 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 폐기물로 분류되어 매립 또는 소각된다.

따라서 쓰레기 사진을 통해 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용할 수 있는 시스템이 필요하다. 본 프로젝트에서는 일반 쓰레기, 플라스틱, 종이, 유리 등 10종류의 쓰레기를 분류하는 **Object Detection**을 수행한다.

</br>

# 🗂️ 데이터셋

- **데이터의 개수** : Train 4,883장 / Test 4,871장

- **이미지 크기** : (1024, 1024)

- **클래스의 분류**

  - General trash
  - Paper
  - Paper pack
  - Metal
  - Glass
  - Plastic
  - Styrofoam
  - Plastic bag
  - Battery
  - Clothing

</br>

# 🏆 프로젝트 결과

학습한 상위 7개의 모델의 inference 결과를 WBF(Weighted Boxes Function)을 사용하여 제출한 inference 결과가 가장 높은 score를 달성.

- **앙상블에 사용한 모델**


  | idx  |                  model                 |  mAP 50  |
  | :--: | :------------------------------------: | :------: |
  |  1   |   SwinTransformer Large - FasterRCNN   |  0.5893  |
  |  2   |   SwinTransformer Large - FasterRCNN   |  0.5949  |
  |  3   |   SwinTransformer Large - FasterRCNN   |  0.6072  |
  |  4   |   SwinTransformer Large - FasterRCNN   |  0.6100  |
  |  5   |   SwinTransformer Large - FasterRCNN   |  0.6176  |
  |  6   |                 Yolov7                 |  0.5633  |
  |  7   |              EfficientDet              |  0.5556  |

- **최종 결과**

  mAP : 0.6753
 
  <img width="821" alt="Untitled" src="https://user-images.githubusercontent.com/74412423/206113824-5632adb4-3e84-4b25-bfe3-d811cbec9423.png">


</br>

# 👨‍👨‍👧‍👧 팀 구성 및 역할

<table>
  <tr height="35px">
    <td align="center" width="180px">
      <a href="https://github.com/miinngdok">김동인_T4029</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/shinunjune">신원준_T4113</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/iihye">이혜진_T4177</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/22eming">정혁기_T4205</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/jun981015">홍준형_T4235</a>
    </td>
  </tr>
  <tr height="35px">
    <td align="center" width="180px">
      <a> Model experiments, Ensemble </a>
    </td>
    <td align="center" width="180px">
      <a> EDA, Data Augmentation </a>
    </td>
    <td align="center" width="180px">
      <a> Model experiments </a>
    </td>
    <td align="center" width="180px">
      <a> EDA, Data Augmentation </a>
    </td>
    <td align="center" width="180px">
      <a> Model experiments, Data Augmentation </a>
    </td>
  </tr>
</table>
