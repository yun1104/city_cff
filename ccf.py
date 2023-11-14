import torch
from torch import nn
import torch.nn as nn
import numpy as np
import pandas as pd


submit_path = 'submission_sample.csv'
submit_data = pd.read_csv(submit_path)

density_path = '人口密度.xlsx'
density_data = pd.read_excel(density_path)
# 人口密度.xlsx没有缺失值

scale_path = '人口规模.xlsx'
scale_data = pd.read_excel(scale_path)


Urbanization_path = '城镇化率.xlsx'
Urbanization_data = pd.read_excel(Urbanization_path)
# 没有缺失

work_path = '就业信息.xlsx'
work_data = pd.read_excel(work_path)
# 有多个表，但只有城镇失业率有缺失

salary_path = '工资水平.xlsx'
salary_data = pd.read_excel(salary_path)

age_path = '年龄结构.xlsx'
age_data = pd.read_excel(age_path)
# 有缺失
life_path = '生活水平.xlsx'
life_data = pd.read_excel(life_path)

# 有多个表,多个表有缺失










