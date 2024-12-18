import pandas as pd

# Dữ liệu HOTA
hota_data = {
    'Sequence': ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP', 'COMBINED'],
    'HOTA': [61.681, 84.537, 66.549, 74.709, 56.995, 70.578, 67.384, 73.65],
    'DetA': [63.352, 84.296, 69.543, 78.031, 61.152, 72.044, 70.364, 74.024],
    'AssA': [60.544, 84.878, 64.138, 71.598, 53.462, 69.358, 64.714, 73.669],
    'DetRe': [71.392, 88.946, 76.055, 82.076, 66.182, 80.34, 75.785, 80.232],
    'DetPr': [77.44, 89.545, 81.167, 87.159, 77.387, 82.712, 81.066, 84.287],
    'AssRe': [65.991, 88.692, 71.689, 77.09, 58.351, 74.673, 75.541, 79.076],
    'AssPr': [78.753, 90.41, 79.424, 84.863, 74.606, 86.739, 73.053, 84.93],
    'LocA': [84.941, 90.383, 85.944, 87.869, 81.865, 89.08, 84.441, 87.635],
    'RHOTA': [65.686, 86.88, 69.779, 76.645, 59.455, 74.635, 70.028, 76.85],
    'HOTA(0)': [75.644, 94.644, 79.761, 87.787, 74.965, 80.476, 83.651, 86.01],
    'LocA(0)': [80.32, 89.226, 82.989, 85.959, 76.902, 87.244, 81.088, 85.017],
    'HOTALocA(0)': [60.757, 84.447, 66.192, 75.461, 57.65, 70.21, 67.831, 73.123]
}

# Dữ liệu CLEAR
clear_data = {
    'Sequence': ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP', 'COMBINED'],
    'MOTA': [72.565, 94.91, 82.438, 91.251, 76.47, 80.089, 86.825, 86.08],
    'MOTP': [83.111, 89.485, 84.164, 86.417, 79.326, 87.91, 82.461, 86.163],
    'MODA': [73.045, 94.952, 83.22, 91.383, 77.054, 80.779, 87.178, 86.393],
    'CLR_Re': [82.617, 97.141, 88.461, 92.775, 81.288, 88.956, 90.332, 90.791],
    'CLR_Pr': [89.616, 97.796, 94.407, 98.521, 95.05, 91.582, 96.626, 95.38],
    'MTR': [57.627, 92.771, 67.424, 92.308, 64.912, 69.863, 79.091, 73.889],
    'PTR': [35.593, 6.0241, 23.485, 7.6923, 31.579, 17.808, 14.545, 19.63],
    'MLR': [6.7797, 1.2048, 9.0909, 0, 3.5088, 12.329, 6.3636, 6.4815],
    'sMOTA': [58.612, 84.695, 68.429, 78.649, 59.664, 69.334, 70.982, 73.518],
    'CLR_TP': [15328, 46147, 6110, 4931, 10426, 8377, 10511, 101830],
    'CLR_FN': [3225, 1358, 797, 384, 2400, 1040, 1125, 10329],
    'CLR_FP': [1776, 1040, 362, 74, 543, 770, 367, 4932],
    'IDS': [89, 20, 54, 7, 75, 65, 41, 351],
    'MT': [34, 77, 89, 24, 37, 51, 87, 399],
    'PT': [21, 5, 31, 2, 18, 13, 16, 106],
    'ML': [4, 1, 12, 0, 2, 9, 7, 35],
    'Frag': [129, 43, 51, 10, 199, 48, 71, 551]
}

# Dữ liệu Identity
identity_data = {
    'Sequence': ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP', 'COMBINED'],
    'IDF1': [72.9, 94.631, 79.767, 88.953, 71.275, 80.241, 81.292, 84.785],
    'IDR': [70.053, 94.314, 77.255, 86.359, 66.116, 79.091, 78.644, 82.745],
    'IDP': [75.988, 94.95, 82.447, 91.708, 77.309, 81.426, 84.124, 86.928],
    'IDTP': [12997, 44804, 5336, 4590, 8480, 7448, 9151, 92806],
    'IDFN': [5556, 2701, 1571, 725, 4346, 1969, 2485, 19353],
    'IDFP': [4107, 2383, 1136, 415, 2489, 1699, 1727, 13956]
}

# Tạo DataFrames
hota_df = pd.DataFrame(hota_data)
clear_df = pd.DataFrame(clear_data)
identity_df = pd.DataFrame(identity_data)

# Lưu vào Excel
with pd.ExcelWriter('tracking_evaluation_results.xlsx', engine='openpyxl') as writer:
    hota_df.to_excel(writer, sheet_name='HOTA', index=False)
    clear_df.to_excel(writer, sheet_name='CLEAR', index=False)
    identity_df.to_excel(writer, sheet_name='Identity', index=False)

print("Dữ liệu đã được lưu vào 'tracking_evaluation_results.xlsx'.")
