import torch
from torch import nn
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

submit_path = 'submission_sample.csv'
submit_data = pd.read_csv(submit_path)

density_path = '人口密度.xlsx'
density_data = pd.read_excel(density_path)

scale_path = '人口规模.xlsx'
scale_data = pd.read_excel(scale_path)

urbanization_path = '城镇化率.xlsx'
urbanization_data = pd.read_excel(urbanization_path)

work_path = '就业信息.xlsx'
work1_data = pd.read_excel(work_path, '城镇失业率')
work2_data = pd.read_excel(work_path, '从业人员数')
work3_data = pd.read_excel(work_path, '第一、二、三产业就业人数')

salary_path = '工资水平.xlsx'
salary_data = pd.read_excel(salary_path)
salary_melted = salary_data.melt(id_vars='averageWage', var_name='city', value_name='value')
salary_melted = salary_melted[['city', 'averageWage', 'value']]
salary_melted = salary_melted.rename(columns={'city': '城市名称', 'averageWage': '年份', 'value': 'averageWage'})
salary_melted['年份'] = salary_melted['年份'].str.replace('年', '').astype('int64')

# age_path = '年龄结构.xlsx'
# age_data = pd.read_excel(age_path).rename(columns={'城市': '城市名称'})

# life_path = '生活水平.xlsx'
# life_data = pd.read_excel(life_path)
# life_avg_in = pd.read_excel(life_path, '人均可支配收入', header=1)
# life_avg_out = pd.read_excel(life_path, '人均消费支出', header=1)
# life_urban_avg_in = pd.read_excel(life_path, '城镇居民人均收入', header=1)
# life_urban_avg_out = pd.read_excel(life_path, '城镇居民消费支出', header=1)
# life_rural_avg_in = pd.read_excel(life_path, '农村居民人均收入', header=1)
# life_rural_avg_out = pd.read_excel(life_path, '农村居民消费支出', header=1)
# life_tables = [life_avg_in, life_avg_out, life_urban_avg_in, life_urban_avg_out, life_rural_avg_in, life_rural_avg_out]
# life_varname = ['人均可支配收入', '人均消费支出', '城镇居民人均收入', '城镇居民消费支出', '农村居民人均收入', '农村居民消费支出']
# for i in range(len(life_tables)):
#     life_tables[i] = life_tables[i].melt(id_vars='城市名称', var_name='年份', value_name=life_varname[i])

# 合并
data_tables = [scale_data, density_data, urbanization_data, work1_data, work2_data, work3_data, salary_melted]
# data_tables.extend(life_tables)
for i in range(len(data_tables)):
    data_tables[i]['城市名称'] = data_tables[i]['城市名称'].str.replace(' ', '')
    data_tables[i]['城市名称'] = data_tables[i]['城市名称'].str.replace('ctiy', 'city')
    data_tables[i]['城市名称'] = data_tables[i]['城市名称'].str.replace('city', '')
    data_tables[i]['城市名称'] = data_tables[i]['城市名称'].astype(int)
data_table = reduce(lambda left, right: pd.merge(left, right, on=['城市名称', '年份'], how='outer'), data_tables)

# 去重
data_table = data_table.drop_duplicates(subset=['城市名称', '年份'])

# 排序
data_table = data_table.sort_values(by=['城市名称', '年份'])

data_table.to_excel('merged_file.xlsx', index=False)

df = pd.read_excel("merged_file.xlsx")

# 根据条件筛选出需要删除的行
rows_to_delete = df[(df["年份"] < 2006) | (df["年份"] == 2022)]

# 删除符合条件的行
df = df.drop(rows_to_delete.index)

# 保存修改后的数据到新的Excel文件
df.to_excel("new_modified_file.xlsx", index=False)

real_data = pd.read_excel('new_modified_file.xlsx')
# 缺失值处理
grouped_data = real_data.groupby(['城市名称'])
temp_data = pd.DataFrame()
temp_list = []
for name ,data in grouped_data:
    for column in data.columns[1:]:
        data[column] = data[column].fillna(method='bfill').fillna(method='ffill')
    data = data.dropna(axis=1, how='any')
    temp_list.append(data)
# 缺失值处理
#异常值处理

new_temp_list = []
for group_df in temp_list:
    for column in group_df.columns:
        # group_df[column] = group_df[column].replace(0, method='bfill').replace(0, method='ffill')
        # zero_cols = group_df.columns[(group_df == 0).any()]
        # group_df = group_df.drop(zero_cols, axis=1)
        value = group_df[column]
        value_mean = value.mean()
        value_std = value.std()
        outliers = (value > value_mean + 3 * value_std) | (value < value_mean - 3 * value_std)
        if outliers.iloc[0]:
            group_df[column] = group_df[column].bfill()
        elif outliers.iloc[-1]:
            group_df[column] = group_df[column].ffill()
        # scaled_values = scaler.fit_transform(group_df[[column]])
        # group_df[column] = scaled_values.flatten()
    new_temp_list.append(group_df)
# 异常值处理
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.4)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size),
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.hidden_size)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


def train_model(model, X_train, y_train, num_epochs):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.006)
    best_train_loss = float('inf')
    counter = 0
    patience = 200
    avg_loss_list = []
    for t in range(num_epochs):
        total_loss = 0
        for i in range(len(X_train)):
            x = X_train[i]
            y = y_train[i]
            x_tensor = torch.Tensor(x).unsqueeze(0)
            y_tensor = torch.Tensor(y).unsqueeze(0)
            model.hidden = model.reset_hidden_state()
            y_pred = model(x_tensor)
            temp_pred = y_pred * (max_val[1] - min_val[1]) + min_val[1]
            temp_train = y_tensor * (max_val[1] - min_val[1]) + min_val[1]
            loss = loss_fn(temp_pred.float().to(device), temp_train)
            total_loss += loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        avg_loss = total_loss / len(X_train)
        avg_loss_list.append(avg_loss)
        if t % 10 == 0:
            print(f'Epoch {t} train loss: {avg_loss}')
            print('------')
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience :
                print(f'Early stopping at epoch {t}')
                break
    plt.plot(range(t+1), avg_loss_list)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    return model.eval()


# 创建滑动窗口序列
def create_sliding_windows(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size - 1):
        window = data[i:i + window_size]
        X.append(window)
        y.append(data[i + window_size][1])
    return np.array(X), np.array(y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
window_size = 6
scaler = MinMaxScaler()
a = 0
res = []
for data in new_temp_list:
    data = data.iloc[:, 1:]
    a= a+1
    print('我是city:',a)
    scaled_values = scaler.fit_transform(data)
    min_val = scaler.data_min_
    max_val = scaler.data_max_
    data = pd.DataFrame(scaled_values, columns=data.columns)
    X_train, y_train = create_sliding_windows(data.values, window_size)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    input_size = len(data.columns)  # 输入特征维度
    num_layers = 2
    hidden_size = 13
    num_epochs = 1000
    model = LSTMModel(input_size, hidden_size, num_layers, window_size)
    model = model.to(device)
    X_train = X_train.to(device)  # 将训练数据移动到GPU设备上
    y_train = y_train.to(device)
    model = train_model(model, X_train, y_train, num_epochs)
    with torch.no_grad():
        model = model.cpu()
        model.hidden = (model.hidden[0].cpu(), model.hidden[1].cpu())
        test_seq = X_train[-1:].cpu()
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, window_size, input_size).float()
        preds_inverse = pred * (max_val[1] - min_val[1]) + min_val[1]
        # predicted_cases = np.expand_dims(pred, axis=0).flatten()
        print(preds_inverse)
        res.append(round(preds_inverse))
print(res)


submit_data['pred'] =res
# 保存修改后的数据到新的CSV文件
submit_data.to_csv('new_submission_sample.csv', index=False)
# #






#





