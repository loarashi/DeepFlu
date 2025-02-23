DeepFlu
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DeepFlu模型，一層輸入層(kernel_initializer使用uniform, activation使用relu)，四層隱藏層100個節點(kernel_initializer使用uniform, activation使用relu)，一層dropout參數設定0.1，一層輸出層(kernel_initializer使用uniform, activation使用sigmoid)；loss使用binary_crossentropy，optimizer使用adam，validation_spli設定0.1，epochs設定150，batch_size設定200。


<div align=center><img width="600" height="600" src="https://github.com/loarashi/DeepFlu/blob/main/DeepFlu%20model1.png"></div>
 

Dataset link
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. GSE52428 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52428
2. GSE73072 : https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73072

Description:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* Best Learning Method for the Gene Expression Data
  
  將data中的alltime_data.txt檔，帶入deepflu_4layer_alltime.py檔，GSE52428資料集H1N1與H3N2受試者的所有時間點血液樣本，隨機選取80%為訓練資料，20%為測試資料，使用deepflu模型進行預測，可顯示預測結果的Accuracy、Sensitivity、Specificity、Precition、MCC、AUROC、AUPR。

* Time Spans on Prediction Performance
  
  將data中的t0_data.txt檔，帶入deepflu_4layer_t0.py檔，GSE52428資料集H1N1與H3N2受試者的接種病毒前時間點血液樣本，進行leave one out模式，提取一名受試者為測試資料，剩下的受試者為訓練資料，對所有受試者預測一輪，使用deepflu模型進行預測，可得出原始標籤、接種前兩個時間點的預測標籤，接種前兩個時間點的標籤機率，利用前兩樣資訊可計算出每個受試者的TP、TN、FP、FN，並統計所有受試者的TP總和以及剩下的三樣數值以此類推。利用總和後的TP、TN、FP、FN算出Accuracy、Sensitivity、Specificity、Precition、MCC。並將接種前兩個時間點的標籤機率，相加除以二產生所有受試者的平均機率，搭配原始標籤，產生出auroc_auprc_data，帶入auroc_auprc.py，計算出一百次的AUROC、AUPR。

* External Validation on DeepFlu
  
  將data中的external_validation_data.txt檔，分別帶入deepflu_external_validation_h1n1.py檔與deepflu_external_validation_h31n2.py，GSE73072資料集H1N1的DEE3、DEE4實驗；H3N2的DEE2、DEE5實驗受試者的所有時間點血液樣本，H1N1選取DEE3為訓練資料，DEE4為測試資料，H3N2選取DEE2為訓練資料，DEE5為測試資料，使用deepflu模型進行預測，可顯示預測結果的Accuracy、Sensitivity、Specificity、Precition、MCC、AUROC、AUPR。
  
  

