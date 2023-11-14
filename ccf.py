import torch
from torch import nn
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

submit_path = 'submission_sample.csv'
submit_data = pd.read_csv(submit_path)

density_path = '人口密度.xlsx'
density_data = pd.read_excel(density_path)
# 人口密度.xlsx没有缺失值

scale_path = '人口规模.xlsx'
scale_data = pd.read_excel(scale_path)
for column in scale_data.columns[1:]:
    value_mean = scale_data[column].mean()
    scale_data[column].fillna(value_mean)

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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5)
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
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(num_epochs):
        model.hidden = model.reset_hidden_state()
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float().to(device), y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

    return model.eval()


# 创建滑动窗口序列
def create_sliding_windows(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size - 1):
        window = data[i:i + window_size]
        X.append(window)
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
window_size = 6
scaler = MinMaxScaler()
res = {}
group_data = scale_data.groupby(['城市名称'])
for name, data in group_data:
    data = data.iloc[:, :-1]
    data = data.iloc[:, 1:]
    X_train, y_train = create_sliding_windows(data.values, window_size)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    input_size = 2  # 输入特征维度
    num_layers = 1
    hidden_size = 128
    num_epochs = 500
    model = LSTMModel(input_size, hidden_size, num_layers, window_size)
    model = model.to(device)
    X_train = X_train.to(device)  # 将训练数据移动到GPU设备上
    y_train = y_train.to(device)
    model = train_model(model, X_train, y_train, num_epochs)
    with torch.no_grad():
        model = model.cpu()
        model.hidden = (model.hidden[0].cpu(), model.hidden[1].cpu())
        test_seq = X_train[-1:].cpu()
        preds = []
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        print(new_seq)
        # test_seq = torch.as_tensor(new_seq).view(1, window_size, 1).float()
        # predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()
        # print(preds, predicted_cases)
        # res[name] = predicted_cases
