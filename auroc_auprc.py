# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:24:56 2020

@author: anna
"""

import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
all_df = pd.read_csv("data/auroc_auprc_data/H1N1_auroc_auprc_no237.txt",sep='\t',encoding='utf-8')#讀取H1N1_auroc_auprc_no237.txt檔案

l = all_df.values[:,0]#匯入標籤
p1 = all_df.values[:,1]#匯入第1次數值，以下為2~100次數值
p2 =all_df.values[:,2]
p3 =all_df.values[:,3]
p4 =all_df.values[:,4]
p5 =all_df.values[:,5]
p6 =all_df.values[:,6]
p7 =all_df.values[:,7]
p8 =all_df.values[:,8]
p9 =all_df.values[:,9]
p10 =all_df.values[:,10]
p11 =all_df.values[:,11]
p12 =all_df.values[:,12]
p13 =all_df.values[:,13]
p14 =all_df.values[:,14]
p15 =all_df.values[:,15]
p16 =all_df.values[:,16]
p17 =all_df.values[:,17]
p18 =all_df.values[:,18]
p19 =all_df.values[:,19]
p20 =all_df.values[:,20]
p21 =all_df.values[:,21]
p22 =all_df.values[:,22]
p23 =all_df.values[:,23]
p24 =all_df.values[:,24]
p25 =all_df.values[:,25]
p26 =all_df.values[:,26]
p27 =all_df.values[:,27]
p28 =all_df.values[:,28]
p29 =all_df.values[:,29]
p30 =all_df.values[:,30]
p31 =all_df.values[:,31]
p32 =all_df.values[:,32]
p33 =all_df.values[:,33]
p34 =all_df.values[:,34]
p35 =all_df.values[:,35]
p36 =all_df.values[:,36]
p37 =all_df.values[:,37]
p38 =all_df.values[:,38]
p39 =all_df.values[:,39]
p40 =all_df.values[:,40]
p41 =all_df.values[:,41]
p42 =all_df.values[:,42]
p43 =all_df.values[:,43]
p44 =all_df.values[:,44]
p45 =all_df.values[:,45]
p46 =all_df.values[:,46]
p47 =all_df.values[:,47]
p48 =all_df.values[:,48]
p49 =all_df.values[:,49]
p50 =all_df.values[:,50]
p51 =all_df.values[:,51]
p52 =all_df.values[:,52]
p53 =all_df.values[:,53]
p54 =all_df.values[:,54]
p55 =all_df.values[:,55]
p56 =all_df.values[:,56]
p57 =all_df.values[:,57]
p58 =all_df.values[:,58]
p59 =all_df.values[:,59]
p60 =all_df.values[:,60]
p61 =all_df.values[:,61]
p62 =all_df.values[:,62]
p63 =all_df.values[:,63]
p64 =all_df.values[:,64]
p65 =all_df.values[:,65]
p66 =all_df.values[:,66]
p67 =all_df.values[:,67]
p68 =all_df.values[:,68]
p69 =all_df.values[:,69]
p70 =all_df.values[:,70]
p71 =all_df.values[:,71]
p72 =all_df.values[:,72]
p73 =all_df.values[:,73]
p74 =all_df.values[:,74]
p75 =all_df.values[:,75]
p76 =all_df.values[:,76]
p77 =all_df.values[:,77]
p78 =all_df.values[:,78]
p79 =all_df.values[:,79]
p80 =all_df.values[:,80]
p81 =all_df.values[:,81]
p82 =all_df.values[:,82]
p83 =all_df.values[:,83]
p84 =all_df.values[:,84]
p85 =all_df.values[:,85]
p86 =all_df.values[:,86]
p87 =all_df.values[:,87]
p88 =all_df.values[:,88]
p89 =all_df.values[:,89]
p90 =all_df.values[:,90]
p91 =all_df.values[:,91]
p92 =all_df.values[:,92]
p93 =all_df.values[:,93]
p94 =all_df.values[:,94]
p95 =all_df.values[:,95]
p96 =all_df.values[:,96]
p97 =all_df.values[:,97]
p98 =all_df.values[:,98]
p99 =all_df.values[:,99]
p100 =all_df.values[:,100]

fpr, tpr, thresholds = metrics.roc_curve(l,p1)#計算出第1次fpr&tpr
auc_roc1 = metrics.auc(fpr, tpr)#計算出第1次AUROC，以下以此類推
fpr, tpr, thresholds = metrics.roc_curve(l,p2)
auc_roc2 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p3)
auc_roc3 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p4)
auc_roc4 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p5)
auc_roc5 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p6)
auc_roc6 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p7)
auc_roc7 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p8)
auc_roc8 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p9)
auc_roc9 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p10)
auc_roc10 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p11)
auc_roc11 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p12)
auc_roc12 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p13)
auc_roc13 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p14)
auc_roc14 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p15)
auc_roc15 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p16)
auc_roc16 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p17)
auc_roc17 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p18)
auc_roc18 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p19)
auc_roc19 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p20)
auc_roc20 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p21)
auc_roc21 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p22)
auc_roc22 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p23)
auc_roc23 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p24)
auc_roc24 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p25)
auc_roc25 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p26)
auc_roc26 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p27)
auc_roc27 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p28)
auc_roc28 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p29)
auc_roc29 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p30)
auc_roc30 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p31)
auc_roc31 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p32)
auc_roc32 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p33)
auc_roc33 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p34)
auc_roc34 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p35)
auc_roc35 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p36)
auc_roc36 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p37)
auc_roc37 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p38)
auc_roc38 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p39)
auc_roc39 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p40)
auc_roc40 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p41)
auc_roc41 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p42)
auc_roc42 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p43)
auc_roc43 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p44)
auc_roc44 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p45)
auc_roc45 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p46)
auc_roc46 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p47)
auc_roc47 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p48)
auc_roc48 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p49)
auc_roc49 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p50)
auc_roc50 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p51)
auc_roc51 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p52)
auc_roc52 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p53)
auc_roc53 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p54)
auc_roc54 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p55)
auc_roc55 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p56)
auc_roc56 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p57)
auc_roc57 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p58)
auc_roc58 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p59)
auc_roc59 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p60)
auc_roc60 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p61)
auc_roc61 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p62)
auc_roc62 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p63)
auc_roc63 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p64)
auc_roc64 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p65)
auc_roc65 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p66)
auc_roc66 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p67)
auc_roc67 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p68)
auc_roc68 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p69)
auc_roc69 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p70)
auc_roc70 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p71)
auc_roc71 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p72)
auc_roc72 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p73)
auc_roc73 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p74)
auc_roc74 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p75)
auc_roc75 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p76)
auc_roc76 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p77)
auc_roc77 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p78)
auc_roc78 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p79)
auc_roc79 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p80)
auc_roc80 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p81)
auc_roc81 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p82)
auc_roc82 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p83)
auc_roc83 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p84)
auc_roc84 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p85)
auc_roc85 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p86)
auc_roc86 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p87)
auc_roc87 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p88)
auc_roc88 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p89)
auc_roc89 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p90)
auc_roc90 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p91)
auc_roc91 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p92)
auc_roc92 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p93)
auc_roc93 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p94)
auc_roc94 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p95)
auc_roc95 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p96)
auc_roc96 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p97)
auc_roc97 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p98)
auc_roc98 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p99)
auc_roc99 = metrics.auc(fpr, tpr)
fpr, tpr, thresholds = metrics.roc_curve(l,p100)
auc_roc100 = metrics.auc(fpr, tpr)

precision, recall, thresholds = precision_recall_curve(l,p1)#計算出第1次precision&recall
auc_pr1 = metrics.auc(recall, precision)#計算出第1次AUPR，以下以此類推
precision, recall, thresholds = precision_recall_curve(l,p2)
auc_pr2 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p3)
auc_pr3 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p4)
auc_pr4 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p5)
auc_pr5 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p6)
auc_pr6 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p7)
auc_pr7 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p8)
auc_pr8 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p9)
auc_pr9 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p10)
auc_pr10 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p11)
auc_pr11 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p12)
auc_pr12 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p13)
auc_pr13 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p14)
auc_pr14 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p15)
auc_pr15 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p16)
auc_pr16 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p17)
auc_pr17 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p18)
auc_pr18 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p19)
auc_pr19 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p20)
auc_pr20 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p21)
auc_pr21 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p22)
auc_pr22 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p23)
auc_pr23 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p24)
auc_pr24 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p25)
auc_pr25 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p26)
auc_pr26 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p27)
auc_pr27 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p28)
auc_pr28 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p29)
auc_pr29 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p30)
auc_pr30 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p31)
auc_pr31 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p32)
auc_pr32 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p33)
auc_pr33 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p34)
auc_pr34 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p35)
auc_pr35 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p36)
auc_pr36 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p37)
auc_pr37 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p38)
auc_pr38 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p39)
auc_pr39 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p40)
auc_pr40 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p41)
auc_pr41 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p42)
auc_pr42 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p43)
auc_pr43 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p44)
auc_pr44 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p45)
auc_pr45 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p46)
auc_pr46 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p47)
auc_pr47 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p48)
auc_pr48 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p49)
auc_pr49 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p50)
auc_pr50 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p51)
auc_pr51 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p52)
auc_pr52 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p53)
auc_pr53 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p54)
auc_pr54 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p55)
auc_pr55 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p56)
auc_pr56 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p57)
auc_pr57 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p58)
auc_pr58 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p59)
auc_pr59 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p60)
auc_pr60 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p61)
auc_pr61 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p62)
auc_pr62 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p63)
auc_pr63 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p64)
auc_pr64 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p65)
auc_pr65 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p66)
auc_pr66 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p67)
auc_pr67 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p68)
auc_pr68 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p69)
auc_pr69 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p70)
auc_pr70 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p71)
auc_pr71 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p72)
auc_pr72 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p73)
auc_pr73 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p74)
auc_pr74 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p75)
auc_pr75 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p76)
auc_pr76 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p77)
auc_pr77 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p78)
auc_pr78 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p79)
auc_pr79 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p80)
auc_pr80 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p81)
auc_pr81 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p82)
auc_pr82 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p83)
auc_pr83 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p84)
auc_pr84 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p85)
auc_pr85 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p86)
auc_pr86 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p87)
auc_pr87 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p88)
auc_pr88 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p89)
auc_pr89 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p90)
auc_pr90 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p91)
auc_pr91 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p92)
auc_pr92 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p93)
auc_pr93 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p94)
auc_pr94 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p95)
auc_pr95 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p96)
auc_pr96 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p97)
auc_pr97 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p98)
auc_pr98 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p99)
auc_pr99 = metrics.auc(recall, precision)
precision, recall, thresholds = precision_recall_curve(l,p100)
auc_pr100 = metrics.auc(recall, precision)

print(auc_roc1,auc_pr1)#列印出第1次AUROC&AUPR，以下以此類推
print(auc_roc2,auc_pr2)
print(auc_roc3,auc_pr3)
print(auc_roc4,auc_pr4)
print(auc_roc5,auc_pr5)
print(auc_roc6,auc_pr6)
print(auc_roc7,auc_pr7)
print(auc_roc8,auc_pr8)
print(auc_roc9,auc_pr9)
print(auc_roc10,auc_pr10)
print(auc_roc11,auc_pr11)
print(auc_roc12,auc_pr12)
print(auc_roc13,auc_pr13)
print(auc_roc14,auc_pr14)
print(auc_roc15,auc_pr15)
print(auc_roc16,auc_pr16)
print(auc_roc17,auc_pr17)
print(auc_roc18,auc_pr18)
print(auc_roc19,auc_pr19)
print(auc_roc20,auc_pr20)
print(auc_roc21,auc_pr21)
print(auc_roc22,auc_pr22)
print(auc_roc23,auc_pr23)
print(auc_roc24,auc_pr24)
print(auc_roc25,auc_pr25)
print(auc_roc26,auc_pr26)
print(auc_roc27,auc_pr27)
print(auc_roc28,auc_pr28)
print(auc_roc29,auc_pr29)
print(auc_roc30,auc_pr30)
print(auc_roc31,auc_pr31)
print(auc_roc32,auc_pr32)
print(auc_roc33,auc_pr33)
print(auc_roc34,auc_pr34)
print(auc_roc35,auc_pr35)
print(auc_roc36,auc_pr36)
print(auc_roc37,auc_pr37)
print(auc_roc38,auc_pr38)
print(auc_roc39,auc_pr39)
print(auc_roc40,auc_pr40)
print(auc_roc41,auc_pr41)
print(auc_roc42,auc_pr42)
print(auc_roc43,auc_pr43)
print(auc_roc44,auc_pr44)
print(auc_roc45,auc_pr45)
print(auc_roc46,auc_pr46)
print(auc_roc47,auc_pr47)
print(auc_roc48,auc_pr48)
print(auc_roc49,auc_pr49)
print(auc_roc50,auc_pr50)
print(auc_roc51,auc_pr51)
print(auc_roc52,auc_pr52)
print(auc_roc53,auc_pr53)
print(auc_roc54,auc_pr54)
print(auc_roc55,auc_pr55)
print(auc_roc56,auc_pr56)
print(auc_roc57,auc_pr57)
print(auc_roc58,auc_pr58)
print(auc_roc59,auc_pr59)
print(auc_roc60,auc_pr60)
print(auc_roc61,auc_pr61)
print(auc_roc62,auc_pr62)
print(auc_roc63,auc_pr63)
print(auc_roc64,auc_pr64)
print(auc_roc65,auc_pr65)
print(auc_roc66,auc_pr66)
print(auc_roc67,auc_pr67)
print(auc_roc68,auc_pr68)
print(auc_roc69,auc_pr69)
print(auc_roc70,auc_pr70)
print(auc_roc71,auc_pr71)
print(auc_roc72,auc_pr72)
print(auc_roc73,auc_pr73)
print(auc_roc74,auc_pr74)
print(auc_roc75,auc_pr75)
print(auc_roc76,auc_pr76)
print(auc_roc77,auc_pr77)
print(auc_roc78,auc_pr78)
print(auc_roc79,auc_pr79)
print(auc_roc80,auc_pr80)
print(auc_roc81,auc_pr81)
print(auc_roc82,auc_pr82)
print(auc_roc83,auc_pr83)
print(auc_roc84,auc_pr84)
print(auc_roc85,auc_pr85)
print(auc_roc86,auc_pr86)
print(auc_roc87,auc_pr87)
print(auc_roc88,auc_pr88)
print(auc_roc89,auc_pr89)
print(auc_roc90,auc_pr90)
print(auc_roc91,auc_pr91)
print(auc_roc92,auc_pr92)
print(auc_roc93,auc_pr93)
print(auc_roc94,auc_pr94)
print(auc_roc95,auc_pr95)
print(auc_roc96,auc_pr96)
print(auc_roc97,auc_pr97)
print(auc_roc98,auc_pr98)
print(auc_roc99,auc_pr99)
print(auc_roc100,auc_pr100)
