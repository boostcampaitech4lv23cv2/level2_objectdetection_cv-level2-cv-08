# π νλ‘μ νΈ κ°μ

<img width="809" alt="Untitled" src="https://user-images.githubusercontent.com/74412423/206112344-eff89532-32cc-49f1-b5ad-dece36f22aa0.png">


λλ μμ°, λλ μλΉμ μλμμ βμ°λ κΈ° λλβ, βλ§€λ¦½μ§ λΆμ‘±βκ³Ό κ°μ μ¬ν λ¬Έμ κ° λ°μνκ³  μλ€. μ΄λ¬ν λ¬Έμ λ₯Ό ν΄κ²°νκΈ° μν΄μ λΆλ¦¬μκ±°λ νκ²½ λΆλ΄μ μ€μΌ μ μλ λ°©λ²μ΄λ€. μ λΆλ¦¬μκ±° λ μ°λ κΈ°λ μμμΌλ‘μ κ°μΉλ₯Ό μΈμ λ°μ μ¬νμ©λμ§λ§, μλͺ» λΆλ¦¬λ°°μΆ λλ©΄ νκΈ°λ¬Όλ‘ λΆλ₯λμ΄ λ§€λ¦½ λλ μκ°λλ€.

λ°λΌμ μ°λ κΈ° μ¬μ§μ ν΅ν΄ μ°λ κΈ°μ₯μ μ€μΉλμ΄ μ νν λΆλ¦¬μκ±°λ₯Ό λκ±°λ, μ΄λ¦°μμ΄λ€μ λΆλ¦¬μκ±° κ΅μ‘ λ±μ μ¬μ©ν  μ μλ μμ€νμ΄ νμνλ€. λ³Έ νλ‘μ νΈμμλ μΌλ° μ°λ κΈ°, νλΌμ€ν±, μ’μ΄, μ λ¦¬ λ± 10μ’λ₯μ μ°λ κΈ°λ₯Ό λΆλ₯νλ **Object Detection**μ μννλ€.

</br>

# ποΈ λ°μ΄ν°μ

- **λ°μ΄ν°μ κ°μ** : Train 4,883μ₯ / Test 4,871μ₯

- **μ΄λ―Έμ§ ν¬κΈ°** : (1024, 1024)

- **ν΄λμ€μ λΆλ₯**

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

# π νλ‘μ νΈ κ²°κ³Ό

νμ΅ν μμ 7κ°μ λͺ¨λΈμ inference κ²°κ³Όλ₯Ό WBF(Weighted Boxes Function)μ μ¬μ©νμ¬ μ μΆν inference κ²°κ³Όκ° κ°μ₯ λμ scoreλ₯Ό λ¬μ±.

- **μμλΈμ μ¬μ©ν λͺ¨λΈ**


  | idx  |                  model                 |  mAP 50  |
  | :--: | :------------------------------------: | :------: |
  |  1   |   SwinTransformer Large - FasterRCNN   |  0.5893  |
  |  2   |   SwinTransformer Large - FasterRCNN   |  0.5949  |
  |  3   |   SwinTransformer Large - FasterRCNN   |  0.6072  |
  |  4   |   SwinTransformer Large - FasterRCNN   |  0.6100  |
  |  5   |   SwinTransformer Large - FasterRCNN   |  0.6176  |
  |  6   |                 Yolov7                 |  0.5633  |
  |  7   |              EfficientDet              |  0.5556  |

- **μ΅μ’ κ²°κ³Ό**

  mAP : 0.6753
 
  <img width="821" alt="Untitled" src="https://user-images.githubusercontent.com/74412423/206113824-5632adb4-3e84-4b25-bfe3-d811cbec9423.png">


</br>

# π¨βπ¨βπ§βπ§ ν κ΅¬μ± λ° μ­ν 

<table>
  <tr height="35px">
    <td align="center" width="180px">
      <a href="https://github.com/miinngdok">κΉλμΈ_T4029</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/shinunjune">μ μμ€_T4113</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/iihye">μ΄νμ§_T4177</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/22eming">μ νκΈ°_T4205</a>
    </td>
    <td align="center" width="180px">
      <a href="https://github.com/jun981015">νμ€ν_T4235</a>
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
