#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'pymarl'
@author  : '张宗旺'
@file    : 'plot'.py
@ide     : 'PyCharm'
@time    : '2020'-'03'-'15' '19':'03':'05'
@contact : zongwang.zhang@outlook.com
'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import json
from matplotlib import rcParams
from matplotlib import rc
import os
rcParams['mathtext.fontset'] = 'stix'
# plt.tight_layout()

def plot_result_a():
    '''
    方差范围变化的影响
    '''
    x = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    #FIM
    y1 = [0.06797375462262951, 0.09647887095877009, 0.13193431510073164, 0.17414914956791394, 0.22379086251227737,
          0.2826553877724272, 0.34029964925855866, 0.4023451938791697, 0.4888402002066754,0.5568794692024963,0.6350702258106339]
    #uniform
    y2 = [0.10381994188061916, 0.15132324404806602, 0.20332748486796692, 0.2680418643178639, 0.3551372856998092,
          0.43797843778912565, 0.5308250245823766, 0.6184836675123626, 0.7654721504134064, 0.8585521724654143,1.0044881744887204]
    #direct
    y3 = [0.025130020627944904, 0.03699438594081933, 0.051328224340018914, 0.06917291884671439, 0.08824466467122315, 0.10711437576296376,
          0.13400281289525467, 0.16613491938435143, 0.17899414077748682, 0.21399514771219952, 0.26713471453209187]
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    max = np.max([np.max(y1),np.max(y2),np.max(y3)])
    y1 = y1/max
    y2 = y2/max
    y3 = y3/max
    plt.figure(figsize=(5.6, 5.6))
    plt.rc('text',usetex=True)
    plt.plot(x, y1, "ro-", label="FIM", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, y2, "cs--", label="Uniform", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, y3, "*--", label="Direct", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("scale of "+ r"$\theta$", fontsize=16)
    plt.ylabel("normalized MSE", fontsize=16)
    plt.yticks(fontsize=16)
    newxticks=['[5,5]','[5,7]','[5,9]','[5,11]','[5,13]','[5,15]',
               '[5,17]','[5,19]','[5,21]','[5,23]','[5,25]']
    plt.xticks(x,newxticks,fontsize=16,rotation=60)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.legend(fontsize=16)
    plt.grid(linestyle='--')
    plt.savefig("MSE-a.eps",dpi=200)
    plt.show()

def plot_result_b():
    '''
    均匀比例k步逼近最优比例
    '''

    x = [0,1,2,3,4,5,6,7,8,9,10]
    # y = [0.7408515035904484, 0.6833765399177655, 0.737392263417109, 0.6452250167487391, 0.6646658520193722,
    #  0.6135340224754267, 0.598226897880374, 0.5250053659214428, 0.5262349489708819, 0.4723124258510271,0.4329851850689412]
    y1 = [0.12487542698801428, 0.11819104410497609, 0.11770546544250735, 0.11910761431295881, 0.1217756353292778, 0.1148307850535631,
          0.1216253216315619, 0.11916469108741709, 0.11724927467785587, 0.1203534915677098, 0.11411184819540392]
    y2 = [0.7186545573040739, 0.7642256038949522, 0.6769935266957436, 0.6678520572309318, 0.6353672496264141, 0.6414237457860746,
          0.5836255324000396, 0.5887684708107802, 0.5514363686311926, 0.49193317954984384, 0.48522754426453496]
    y1 = np.array(y1)
    y2 = np.array(y2)
    max = np.max([np.max(y1), np.max(y2)])
    y1 = y1/max
    y2 = y2/max
    plt.figure(figsize=(5.6,5.6))
    rc('text', usetex=True)
    plt.plot(x, y1, "ro-", linewidth=3, label = "Uniform-to-Direct",
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, y2, "bs-", linewidth=3, label= "Uniform-to-FIM",
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.xlabel("approximation ratio", fontsize=16)
    plt.ylabel("normalized MSE", fontsize=16)
    plt.yticks(fontsize=16)
    # newxticks = ['0%%','10%%','20%%','30%%','40%%','50%%','60%%','70%%','80%%','90%%','100%%']
    newxticks = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']
    plt.xticks(x,newxticks,fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(fontsize=16)
    plt.savefig("MSE-b.eps",dpi=200)
    # plt.savefig("MSE-b.png", dpi=200, bbox_inches='tight')
    plt.show()

def plot_result_c():
    '''
    数据量对 Uniform 和 FIM的影响
    '''
    x = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000,21000]
    #FIM
    y1 =  [2.219224103605739,0.7268422961989344,0.45366128233097663,0.3289214586063162,0.25313499992081434,
          0.205519944815278,0.1701404309161809,0.15748941767644614,0.12931059137820383,0.12143822816250728,0.10664732446516646]
    #Uniform
    y2 = [3.5094855543317025,1.2410742461988287,0.6822059935179536,0.504045652051426,0.37278225946681987,0.33435388354512624,
        0.2772824363777646, 0.22738572692993186, 0.21314872078127364, 0.18107787169683653,0.18187382326478077]

    y3 = [0.872027428265715, 0.3012006363765658, 0.16351193076443368, 0.12476476980373616, 0.10181192586706118,
          0.07815488138316691, 0.06703290502859685, 0.055538971158959165, 0.051768964194291, 0.045216384519001195, 0.04075756235340373]
    '''
    [50000]
    [0.044499664223452105]
    [0.0687257915271872]
    '''

    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    max = np.max([np.max(y1), np.max(y2),np.max(y3)])
    y1 = y1 / max
    y2 = y2 / max
    y3 = y3 / max
    plt.figure(figsize=(5.6, 5.6))
    rc('text', usetex=True)
    plt.plot(x, y1, "ro-", linewidth=3, label="FIM",
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, y2, "cs--", label="Uniform", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, y3, "*--", label="Direct", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel(r"$\psi$", fontsize=16)
    plt.ylabel("normalized MSE", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(x,x,fontsize=16,rotation=50)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.legend(fontsize=16)
    plt.grid(linestyle='--')
    plt.savefig("MSE-c.eps",dpi=200)
    plt.show()

def plot_cdf():
    rc('text', usetex=True)
    #[5,5]
    x1 = [0, 0.023473794548108433, 0.04323416608358083, 0.06299453761905321, 0.0827549091545256, 0.10251528068999799,
     0.12227565222547039, 0.1420360237609428, 0.16179639529641518, 0.18155676683188757, 0.20131713836735995]
   # [5,15]
    x2 =[0, 0.08884768845895166, 0.2288093446948519, 0.3687710009307521, 0.5087326571666524, 0.6486943134025526,
     0.7886559696384527, 0.928617625874353, 1.0685792821102533, 1.2085409383461536, 1.3485025945820537]
    # [5,5]
    x3 = [0, 0.1966162577207473, 0.5453245763909024, 0.8940328950610575, 1.2427412137312126, 1.5914495324013676,
     1.9401578510715227, 2.288866169741678, 2.6375744884118326, 2.986282807081988, 3.334991125752143]
    # [5,10]
    x4 = [0, 0.04950515637221427, 0.11351869921374029, 0.1775322420552663, 0.24154578489679235, 0.3055593277383184, 0.36957287057984445, 0.4335864134213704, 0.49759995626289644, 0.5616134991044225, 0.6256270419459484]

    # [5,20]
    x5 = [0, 0.14929979465670834, 0.4031996644248807, 0.657099534193053, 0.9109994039612255, 1.1648992737293977, 1.4187991434975702, 1.6726990132657424, 1.926598883033915, 2.1804987528020874, 2.434398622570259]

    # [5,30]
    x6 = [0, 0.29301087126089337, 0.8329054022163552, 1.3727999331718173, 1.912694464127279, 2.45258899508274, 2.9924835260382023, 3.5323780569936645, 4.072272587949126, 4.612167118904589, 5.15206164986005]

    max = np.max([np.max(np.array(x1)),np.max(np.array(x2)),np.max(np.array(x3)),np.max(np.array(x4)),np.max(np.array(x5)),np.max(np.array(x6))])
    x1 = np.array(x1)/max
    x2 = np.array(x2)/max
    x3 = np.array(x3)/max
    x4 = np.array(x4)/max
    x5 = np.array(x5)/max
    x6 = np.array(x6)/max

    y1 = [0, 0.1998, 0.2, 0.2004, 0.2182, 0.3937, 0.7302, 0.9416, 0.9922, 0.9995, 1.0]
    y2 = [0, 0.1801, 0.3412, 0.5363, 0.7001, 0.8407, 0.9352, 0.979, 0.995, 0.999, 1.0]
    y3 = [0, 0.2143, 0.4178, 0.5837, 0.7303, 0.8509, 0.9274, 0.9724, 0.9927, 0.9979, 1.0]
    y4 = [0, 0.1787, 0.2283, 0.3979, 0.592, 0.7645, 0.8958, 0.9675, 0.9927, 0.9991, 1.0]
    y5 = [0, 0.2128, 0.4184, 0.6061, 0.7585, 0.8771, 0.955, 0.9885, 0.9979, 0.9997, 1.0]
    y6 = [0, 0.2508, 0.4755, 0.6443, 0.7857, 0.8925, 0.9543, 0.9851, 0.9954, 0.9995, 1.0]


    plt.figure(figsize=(8,6))
    plt.plot(x1, y1, "ro--", label="[5,5]",linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x2, y2, "cs--", label="[5,15]", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x3, y3, "b*--", label="[5,25]", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x4, y4, "v-", label="[5,10]", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x5, y5, ">-", label="[5,20]", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x6, y6, "<-", label="[5,30]", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("normalized MSE", fontsize=16)
    plt.ylabel("CDF", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(fontsize=16)
    plt.savefig("MSE-cdf.eps",dpi=200)
    # plt.savefig("MSE-cdf.png", dpi=200, bbox_inches='tight')
    plt.show()


'''
reward重新画
'''

def plot_light_load():
    rc('text', usetex=True)
    plt.figure(figsize=(5.6, 5.6))
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    rl_max_expectation = [0.6 ,0.6 ,0.6 , 0.59946364, 0.6,        0.6196697, 0.65048485, 0.68221818 ,0.76539999 ,0.85576665 ,0.99999998]
    random_mean_reward = [0.2177207  ,0.24057471 ,0.26551089 ,0.2727024 , 0.32140864 ,0.3430153, 0.43849586 ,0.5375387 , 0.50354356 ,0.54480747, 0.6]
    all_local_max_expectation = [0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
    all_offload_max_expectation = [0.08670567, 0.10760576 ,0.12222622 ,0.15019652, 0.19439544 ,0.24166517,
          0.31058356 ,0.41615737 ,0.56410591, 0.75050855 ,1.]
    plt.plot(x, rl_max_expectation, "ro-", label="QMIX", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, random_mean_reward, "cs--", label="random", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_local_max_expectation, "gx--", label="all local", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_offload_max_expectation, "yd--", label="all offload", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("probability of light load", fontsize=16)
    plt.ylabel("normalized $\\bar{\\psi}$", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(x, fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(fontsize=16)
    plt.savefig("light_load.eps",dpi=200)
    plt.show()

def plot_mid_load():
    rc('text', usetex=True)
    plt.figure(figsize=(5.6, 5.6))
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    rl_max_expectation = []
    random_mean_reward = [1.2868184818481851,1.494806480648065,1.9078859885988595,1.9917701770177014,2.0505370537053706,
                    2.4793837383738357,2.146745874587458,2.7882448244824483,3.1556123612361238,3.3368952895289503,3.6363636363636362]

    all_local_max_expectation = [4.000000000000001,3.999999999999998,4.000000000000001,4.000000000000001,4.0,
                                4.0,3.9999999999999996,3.9999999999999996,3.9999999999999996,3.9999999999999973,4.0]

    all_offload_max_expectation = [0.7775027502750276,0.8089008900890093,0.8721872187218724,1.0658465846584662,1.17985798579858,
                                    1.4267026702670262,1.70927592759276,2.1116161616161615,2.538398839883988,2.995319531953194,3.6363636363636362]
    plt.plot(x, rl_max_expectation, "ro-", label="QMIX", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, random_mean_reward, "cs--", label="random", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_local_max_expectation, "gx--", label="all local", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_offload_max_expectation, "yd--", label="all offload", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("probability of moderate load", fontsize=16)
    plt.ylabel("normalized $\\bar{\\psi}$", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(x, fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(fontsize=16)
    plt.savefig("moderate_load.eps",dpi=200)
    plt.show()

def plot_heavy_load():
    rc('text', usetex=True)
    plt.figure(figsize=(5.6, 5.6))
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    rl_max_expectation = [1.0, 0.9885194641155501, 0.9749347324597979, 0.9644234897844148, 0.9627576678894231, 0.9644234897844148, 0.9644234897844145, 0.9644234897844144, 0.9644234897844142, 0.9644234897844144, 0.9644234897844146]
    random_mean_reward = [0.9152232793282916, 0.7592999018343092, 0.6395483370048838, 0.5629251366940974, 0.39636619070611057, 0.35546456515105934, 0.3745859318668342, 0.26862579655491575, 0.13327476135112318, 0.1983341391759378, 0.09548747423608066]
    all_local_max_expectation = [0.9644234897844148, 0.9644234897844145, 0.9644234897844146, 0.9644234897844148, 0.9644234897844144, 0.9644234897844145, 0.9644234897844145, 0.9644234897844142, 0.9644234897844134, 0.964423489784414, 0.9644234897844146]
    all_offload_max_expectation = [0.922899700640919, 0.6372233616852265, 0.43524685762715326, 0.29935925997974894, 0.19558414814393585, 0.1475076516459819, 0.11490928891371985, 0.09861251884744338, 0.09639894558106148, 0.09548747423608073, 0.09548747423608066]

    plt.plot(x, rl_max_expectation, "ro-", label="QMIX", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, random_mean_reward, "cs--", label="random", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_local_max_expectation, "gx--", label="all local", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_offload_max_expectation, "yd--", label="all offload", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("probability of heavy load", fontsize=16)
    plt.ylabel("normalized $\\bar{\\psi}$", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(x, fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(fontsize=16)
    plt.savefig("heavy_load.eps",dpi=200)
    plt.show()

def plot_scale_light():
    rc('text', usetex=True)
    plt.figure(figsize=(5.6, 5.6))
    x = list(range(2,42,2))
    rl_max_expectation = [0.7299560560363707, 0.8334881689928504, 0.8797187068664997, 0.9083762480134496, 0.9269225201124071, 0.943846554717795, 0.9504717252603041, 0.963441012952113, 0.9663068492374733, 0.968478956871287, 0.9771414047369031, 0.9831912885113938, 0.9781526875082507, 0.9938077636470554, 0.983174276661535, 0.9878003541576835, 1.0, 0.9941442667722481, 0.9976530830564162, 0.9932811371687613]
    random_mean_reward = [0.535167861813269, 0.5762047591472013, 0.7553419731381713, 0.6219480931852773, 0.6006104929122102, 0.6353621660097111, 0.5860798191607239, 0.5807105796497929, 0.628254399652345, 0.8595072438305357, 0.5734277048934321, 0.6278703184187762, 0.5337712764854775, 0.6382224483169767, 0.5668057535039187, 0.6345806441237525, 0.5737787927677159, 0.6028611721292738, 0.5681880622538081, 0.5780011173188306]
    all_local_max_expectation = [0.7299560560363713, 0.7299560560363709, 0.7299560560363713, 0.7299560560363714, 0.729956056036371, 0.7299560560363709, 0.7299560560363713, 0.7299560560363713, 0.7299560560363709, 0.729956056036371, 0.7299560560363706, 0.7299560560363713, 0.7299560560363708, 0.729956056036371, 0.7299560560363709, 0.7299560560363711, 0.7299560560363711, 0.7299560560363716, 0.7299560560363714, 0.7299560560363713]
    all_offload_max_expectation = [0.4456961052305256, 0.565844526473289, 0.627153626561422, 0.6731023922546148, 0.6919197169928318, 0.7101783637347037, 0.7110777185327684, 0.7317350788312188, 0.7239700343120575, 0.734820228760148, 0.7431317452958207, 0.7534177616559503, 0.7550888430277058, 0.7528361220863995, 0.7584073811675186, 0.7726452770259584, 0.7564473153266349, 0.7666109220788155, 0.7667680238877732, 0.7692002596269422]
    max = np.max(np.array([np.max(rl_max_expectation),np.max(random_mean_reward),np.max(all_local_max_expectation),np.max(all_offload_max_expectation)]))
    rl_max_expectation = np.array(rl_max_expectation)/max
    random_mean_reward = np.array(random_mean_reward) / max
    all_local_max_expectation = np.array(all_local_max_expectation) / max
    all_offload_max_expectation = np.array(all_offload_max_expectation) / max
    plt.plot(x, rl_max_expectation, "ro-", label="QMIX", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, random_mean_reward, "cs--", label="random", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_local_max_expectation, "gx--", label="all local", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_offload_max_expectation, "yd--", label="all offload", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("$C_c / C_l$", fontsize=16)
    plt.ylabel("normalized $\\bar{\\psi}$", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(x[0:-1:2],x[0:-1:2], fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(fontsize=16,ncol=2)
    plt.savefig("cc_cl_light.eps",dpi=200,)
    plt.show()


def plot_scale_moderate():
    rc('text', usetex=True)
    plt.figure(figsize=(5.6, 5.6))
    x = list(range(2,42,2))
    rl_max_expectation = [0.9998611303919347, 0.9998611303919349, 0.9998611303919349, 0.9998611303919348, 0.9999722260664683,
     0.9998611303919345, 0.9998611303919349, 0.999861130391935, 0.999861130391935, 0.9998611303919349, 1.0,
     0.9998278016697093, 0.9998611303919351, 0.9998611303919345, 0.9998611303919346, 0.9997045866268721,
     0.9994611859636165, 0.9998611303919349, 0.998169057810588, 0.9998611303919347]
    random_mean_reward = [0.6995774396366873, 0.6016627771848211, 0.7239178008991265, 0.7586415466937373, 0.7639479015202698,
     0.8250412702214511, 0.8439390510127392, 0.7797884808400409, 0.7902618610598297, 0.8956985066869031,
     0.8355191448615251, 0.7932389033583925, 0.790871212539211, 0.8185991743194534, 0.8519672997527209,
     0.800857900288683, 0.8094869568113133, 0.7295907519970892, 0.8565289278108185, 0.8114988624861096]
    all_local_max_expectation = [0.999861130391935, 0.9998611303919349, 0.9998611303919342, 0.9998611303919348, 0.9998611303919346,
     0.9998611303919346, 0.9998611303919344, 0.9998611303919351, 0.9998611303919341, 0.9998611303919346,
     0.9998611303919347, 0.9998611303919348, 0.9998611303919348, 0.9998611303919348, 0.9998611303919352,
     0.9998611303919346, 0.9998611303919354, 0.999861130391935, 0.9998611303919351, 0.9998611303919347]
    all_offload_max_expectation = [0.47170432677688734, 0.5543035012038179, 0.5873142131209466, 0.6163028797399871, 0.6276268510020111,
     0.6424431484948722, 0.648278755605891, 0.6505256521335827, 0.6530491649629497, 0.6586171467871068,
     0.6689172867420097, 0.658906659455682, 0.6612755465485962, 0.6759389961422978, 0.669465479722771,
     0.6698860030370092, 0.66899346881917, 0.6745269640441514, 0.6621729214999733, 0.6848753789928733]
    max = np.max(np.array([np.max(rl_max_expectation),np.max(random_mean_reward),np.max(all_local_max_expectation),np.max(all_offload_max_expectation)]))
    rl_max_expectation = np.array(rl_max_expectation)/max
    random_mean_reward = np.array(random_mean_reward) / max
    all_local_max_expectation = np.array(all_local_max_expectation) / max
    all_offload_max_expectation = np.array(all_offload_max_expectation) / max
    plt.plot(x, rl_max_expectation, "ro-", label="QMIX", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, random_mean_reward, "cs--", label="random", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_local_max_expectation, "gx--", label="all local", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_offload_max_expectation, "yd--", label="all offload", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("$C_c / C_l$", fontsize=16)
    plt.ylabel("normalized $\\bar{\\psi}$", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(x[0:-1:2],x[0:-1:2], fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(fontsize=16,ncol=2)
    plt.savefig("cc_cl_moderate.eps",dpi=200,)
    plt.show()


def plot_scale_heavy():
    rc('text', usetex=True)
    plt.figure(figsize=(5.6, 5.6))
    x = list(range(2,42,2))
    rl_max_expectation =  [0.9999999999999994, 0.9999999999999993, 0.9999999999999997, 0.9999999999999996, 1.0, 0.9999999999999992,
                             0.9999999999999992, 0.9999117647111406, 0.9995263159871093, 0.9999999999999992, 0.9977971019744866,
                             0.9977333321173981, 0.998895062386989, 0.9999999999999993, 0.9999999999999993, 0.9999949494997661,
                             0.9998047619163988, 0.999662162363529, 0.9999102564156044, 0.9983983755111691]
    random_mean_reward = [0.18860317460317466, 0.18519105691056922, 0.18512314597970334, 0.17944855967078197, 0.16039903990399054,
                         0.1688793176520449, 0.22978297872340425, 0.1891976007794422, 0.15721828050789965, 0.6196755508173422,
                         0.5298705488884514, 0.1293557676348548, 0.16020200085142616, 0.15202374524481538, 0.251341638981174,
                         0.22643089776267353, 0.19506941767909505, 0.24599765416385916, 0.5714807524059493, 0.2974348377024918]
    all_local_max_expectation = [0.9999999999999989, 0.9999999999999991, 0.9999999999999997, 0.9999999999999997, 0.9999999999999996,
                                 0.9999999999999997, 0.999999999999999, 0.9999999999999996, 0.9999999999999996, 0.9999999999999991,
                                 0.9999999999999996, 0.9999999999999994, 0.9999999999999993, 0.9999999999999994, 0.9999999999999986,
                                 0.9999999999999993, 0.9999999999999998, 0.9999999999999998, 0.9999999999999993, 0.9999999999999996]
    all_offload_max_expectation = [0.09605555555555556, 0.09884878048780489, 0.09958665105386424, 0.10021399176954729, 0.1000812581258126,
                                0.10135973117791297, 0.10054184397163118, 0.1005012787723784, 0.10029543471939517, 0.10106609808102347,
                                0.10083297265394466, 0.1011624896265561, 0.10134355044699883, 0.1007986255982329, 0.10096988532847488,
                                0.10084848484848481, 0.10101432760787599, 0.10162311896383919, 0.10163254593175856, 0.10150234170670887]
    max = np.max(np.array([np.max(rl_max_expectation),np.max(random_mean_reward),np.max(all_local_max_expectation),np.max(all_offload_max_expectation)]))
    rl_max_expectation = np.array(rl_max_expectation)/max
    random_mean_reward = np.array(random_mean_reward) / max
    all_local_max_expectation = np.array(all_local_max_expectation) / max
    all_offload_max_expectation = np.array(all_offload_max_expectation) / max
    plt.plot(x, rl_max_expectation, "ro-", label="QMIX", linewidth=3,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)
    plt.plot(x, random_mean_reward, "cs--", label="random", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_local_max_expectation, "gx--", label="all local", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.plot(x, all_offload_max_expectation, "yd--", label="all offload", linewidth=2,
             markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=8)
    plt.xlabel("$C_c / C_l$", fontsize=16)
    plt.ylabel("normalized $\\bar{\\psi}$", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(x[0:-1:2],x[0:-1:2], fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.legend(loc=(0.1,0.6),fontsize=16,ncol=2)
    plt.savefig("cc_cl_heavy.eps",dpi=200,)
    plt.show()


'''
*****************************************************
'''
def cal_accumulative_reward(r,index,r_tot,gamma=0.9):
    if index == 0:
        return r
    else:
        return r+gamma*r_tot[index-1]

def plot_reward():
    rc('text', usetex=True)
    times = 500000
    reward_file_path = os.path.join(os.getcwd(),'data',r'train_reward4.txt')
    reward_file = open(reward_file_path,"r")
    rewards = np.loadtxt(reward_file)
    print(rewards)
    r = []
    for i in range(times):
        mean_reward = np.mean(rewards[i * 1:(i + 1) * 1])
        r.append(mean_reward)
    plt.figure(figsize=(4.2, 4.2))
    r = r/np.max(r)
    plt.plot(list(range(times)),r)
    plt.xlabel(r"time steps ($10^3$)", fontsize=16)
    plt.ylabel("normalized $r$", fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks([0,50,100,150,200,250],[0,5,10,15,20,25],fontsize=16)
    plt.xticks(fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.savefig("reward1.eps", dpi=200,bbox_inches='tight')
    plt.show()

def plot_reward1():
    rc('text', usetex=True)
    times = 500000
    reward_file_path = os.path.join(os.getcwd(),'data',r'train_reward2.txt')
    reward_file = open(reward_file_path,"r")
    rewards = np.loadtxt(reward_file)[0:times]
    print(rewards)
    r_tot = []
    for index,r in enumerate(rewards):
        r_tot.append(cal_accumulative_reward(r,index,r_tot))

    plt.figure(figsize=(4.2, 4.2))
    r_tot = r_tot/np.max(r_tot)
    plt.plot(list(range(times))[0:-1:1000],r_tot[0:-1:1000])
    # plt.plot(list(range(times)), r_tot)
    plt.xlabel(r"time steps ($10^3$)", fontsize=16)
    plt.ylabel("normalized cumulative reward $r^{tot}$", fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks([0,50,100,150,200,250],[0,5,10,15,20,25],fontsize=16)
    plt.xticks([100000,200000,300000,400000,500000],[100,200,300,400,500],fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.savefig("reward1.eps", dpi=200,bbox_inches='tight')
    plt.show()



def plot_reward2():
    rc('text', usetex=True)
    times = 500
    error_file_path = os.path.join(os.getcwd(), 'data', r'error1.txt')
    error_file = open(error_file_path, "r")
    error = np.loadtxt(error_file)
    print(error)
    e = []
    for i in range(times):
        e.append(np.mean(error[i * 1000:(i + 1) * 1000]))
    plt.figure(figsize=(4.2, 4.2))
    # e = e / np.max(e)
    plt.plot(list(range(times)), e)
    plt.xlabel(r"time steps ($10^3$)", fontsize=16)
    plt.ylabel("average relative error of $r$", fontsize=16)
    plt.yticks([0.05,0.10,0.15,0.20,0.25,0.30,0.35],["5\%","10\%","15\%","20\%","25\%","30\%","35\%"],fontsize=16)
    # plt.xticks([0,50,100,150,200,250],[0,5,10,15,20,25],fontsize=16)
    plt.xticks(fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.grid(linestyle='--')
    plt.savefig("reward2.eps", dpi=200,bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # plot_result_a()
    # plot_result_b()
    # plot_result_c()
    # plot_cdf()
    # plot_light_load()
    plot_mid_load() #数据缺失
    # plot_heavy_load()
    # plot_scale_light()
    # plot_scale_heavy()
    # plot_reward()
    # cal_cdf_mse()
    # plot_reward()
    plot_reward1()
    # plot_reward2() #数据缺失
    # plot_scale_moderate()
    pass
