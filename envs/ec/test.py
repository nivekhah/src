import numpy as np

# Uniform
ex_reward = [[11.1817125, 8.286685714285719, 10.799999999999995, 6.488051785714288],  # 轻载
             [10.826325000000004, 8.208, 10.8, 6.095925],  # 中载载
             [10.821262500000001, 6.083485714285711, 10.8, 4.308283928571428],  # 重载
             # FIM
             [10.895921852374487, 7.345934728644987, 10.315285556903934, 4.23589054678415],  # 轻载
             [10.416353491387753, 7.684494491380737, 10.315285556903927, 5.948636158393299],  # 中载
             [10.414568043359733, 5.988509265576116, 10.315285556903934, 4.23589054678415],  # 重载
             # Direct
             [9.665945999585693, 7.2539306472007565, 8.207999960601615, 6.63227382372966],  # 轻载
             [8.695466582973365, 7.044276128661041, 8.207999960601617, 6.040551285513974],  # 中载
             [8.702508674239608, 5.955617213272167, 8.207999960601615, 4.410329451125811]]  # 重砸

mse_uniform = [[1.4856656044914034, 2.0688543922263283, 1.4233963068605664, 2.3956089246704346],
               [1.4620011953467449, 1.9204568566494817, 1.5728029947657858, 2.4236546321830743],
               [1.5994099972543356, 2.841404891347437, 1.623589895474917, 3.889878818614009]]

# FIM
mse_fim = [[0.8555849452862293, 1.553922970174927, 1.132559869822512, 2.5380074142984768],
           [1.0109961148052262, 1.238312926866686, 1.0076926980390477, 1.462022298964288],
           [1.2309130114799274, 1.9865895036919643, 0.8350595660060559, 2.5951659838938275]]

# Direct
mse_direct = [[0.35821332495222685, 0.48551146105427967, 0.41970740215076435, 0.5051194107455282],
              [0.46098206666816, 0.5111519908841252, 0.42012662525138467, 0.5452716869314472],
              [0.47363835259620274, 0.6033935482429222, 0.4104863430831881, 0.7430470448893255]]

uniform_psi = [[1999, 1481, 1931, 1160],
               [1935, 1467, 1931, 1089],
               [1934, 1087, 1931, 770]]

fim_psi = [[1948, 1313, 1844, 757],
           [1862, 1373, 1844, 1063],
           [1862, 1070, 1844, 757]]

direct_psi = [[1728, 1297, 1467, 1185],
              [1554, 1259, 1467, 1080],
              [1556, 1064, 1467, 788]]


normalized_uniform_psi = np.divide(uniform_psi, np.max(uniform_psi)).tolist()
normalized_fim_psi = np.divide(fim_psi, np.max(fim_psi)).tolist()
normalized_direct_psi = np.divide(direct_psi, np.max(direct_psi)).tolist()

normalized_uniform_mse = np.divide(mse_uniform, np.max(mse_uniform)).tolist()
normalized_fim_mse = np.divide(mse_fim, np.max(mse_fim)).tolist()
normalized_direct_mse = np.divide(mse_direct, np.max(mse_direct)).tolist()

print("uni  psi")
for item in normalized_uniform_psi:
    print(item)

print("fim psi")
for item in normalized_fim_psi:
    print(item)

print("dir psi")
for item in normalized_direct_psi:
    print(item)

print("uni mse")
for item in normalized_uniform_mse:
    print(item)

print("fim mse")
for item in normalized_fim_mse:
    print(item)

print("dir mse")
for item in normalized_direct_mse:
    print(item)

# res = np.array(ex_reward) * 20 * 8.94
# for item in res.tolist():
#     print(list(map(int, item)))
