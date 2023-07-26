
#%%
import pdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.feature_selection import mutual_info_classif
# !pip install backtrader

# !pip install yfinance
# import yfinance as yf
# !pip install lingam
# import feature_selection
import TSASeriesNet
# import feature_selection
if torch.cuda.is_available():
    # dev = "TPU:0"
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print(device)
#%%
%cd D:\projects\Seminar\lingam\Doc\cods\seriesnet_based
data= pd.read_csv('Data/data.csv')
data.set_index("time_stamp",inplace=True,drop=True)

data_top15= pd.read_csv('Data/top15.csv')
data_top15.set_index("time_stamp",inplace=True,drop=True)

Condition= pd.read_csv('Data/Condition.csv')
Condition.set_index("time_stamp",inplace=True,drop=True)

col = list(data.columns)+ list(Condition.columns)+list(data_top15.columns)
cond = pd.DataFrame(np.hstack((data , Condition)))
cond = pd.DataFrame(np.hstack((cond,data_top15)))
cond.columns =col

# data_top15= pd.read_csv('Data/top15_crypto.csv')
# data_top15.set_index("time_stamp",inplace=True,drop=True)
# cond = data_top15[:-1]
# cond = cond.rename(columns={'BTC-USD': 'price_usd_close'})
# cond = cond.fillna(method='ffill')


test_df = cond.iloc[-200:]
xtest = test_df.price_usd_close
condtest = test_df.drop("price_usd_close",axis=1)
condtest =condtest#[f_col]
ytest = test_df.price_usd_close

cond = cond.iloc[:-200]
Y = cond.price_usd_close
X = cond.price_usd_close
cond = cond.drop("price_usd_close",axis=1)

#%%
"""# scale features"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler
MMScaler = MinMaxScaler()
SScaler = StandardScaler()
condscaler = StandardScaler()

MMScaler_test = MinMaxScaler()
SScaler_test = StandardScaler()
condscaler_test = StandardScaler()

# train and validation data scaler
X_trans = SScaler.fit_transform(pd.DataFrame(np.array(Y).reshape(-1, 1)))#Y.reshape(-1, 1))
condtemp = cond
c_trans = condscaler.fit_transform(condtemp)#[f_col])

y_trans = MMScaler.fit_transform(pd.DataFrame(np.array(Y).reshape(-1, 1)))
print(X_trans.shape)
print(c_trans.shape)
print(y_trans.shape)

# Test data Scaler
X_trans_test = SScaler_test.fit_transform(xtest.values.reshape(-1, 1))
c_trans_test = condscaler_test.fit_transform(condtest)
y_trans_test = MMScaler_test.fit_transform(ytest.values.reshape(-1, 1))

"""# split a multivariate sequence past, future samples (X and y)"""
def split_sequences(input_sequences, condition_seq, output_sequence, n_steps_in, n_steps_out):
    X, C, y = list(),list(), list()
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix >= len(input_sequences): break
        seq_x, seq_c, seq_y = input_sequences[i:end_ix], condition_seq[i:end_ix], output_sequence[end_ix:out_end_ix]
        X.append(seq_x), C.append(seq_c) ,y.append(seq_y)
    return np.array(X), np.array(C), np.array(y)


x_shape = 30
y_shape = 7
# Train data
X_ss, C_ss, y_mm = split_sequences(X_trans, c_trans, y_trans, x_shape, y_shape)
print(X_ss.shape, C_ss.shape, y_mm.shape)

#Test data
DataX_test, DataC_test, Datay_test = split_sequences(X_trans_test, c_trans_test, y_trans_test, x_shape, y_shape)

total_samples = len(cond)
train_test_cutoff = round(0.8 * total_samples) # 0.8

X_train = X_ss[:-(total_samples-train_test_cutoff)]
C_train = C_ss[:-(total_samples-train_test_cutoff)]
X_test = X_ss[-(total_samples-train_test_cutoff):]
C_test = C_ss[-(total_samples-train_test_cutoff):]

y_train = y_mm[:-(total_samples-train_test_cutoff)]
y_test = y_mm[-(total_samples-train_test_cutoff):]

# Randomly shuffle the indices
indices = np.arange(len(X_train))
np.random.shuffle(indices)

# Reorder the data using the shuffled indices
X_train = X_train[indices]
C_train = C_train[indices]
y_train = y_train[indices]

print(total_samples - train_test_cutoff)
print("Training Shape:", X_train.shape, y_train.shape)
print("Validatin Shape:", X_test.shape, y_test.shape)

"""# convert to pytorch tensors"""
X_train_tensors = Variable(torch.Tensor(X_train).to(device))
X_test_tensors = Variable(torch.Tensor(X_test).to(device))

C_train_tensors = Variable(torch.Tensor(C_train).to(device))
C_test_tensors = Variable(torch.Tensor(C_test).to(device))

y_train_tensors = Variable(torch.Tensor(y_train).to(device))
y_test_tensors = Variable(torch.Tensor(y_test).to(device))

print(X_train_tensors.shape)
print(C_train_tensors.shape)

# DataX_test, DataC_test, Datay_test
DataX_test = Variable(torch.Tensor(DataX_test).to(device))
DataC_test = Variable(torch.Tensor(DataC_test).to(device))
Datay_test = Variable(torch.Tensor(Datay_test).to(device))

"""# reshaping to rows, timestamps, features"""

X_train_tensors_final = torch.reshape(X_train_tensors,
                                      (X_train_tensors.shape[0], x_shape,
                                       X_train_tensors.shape[2]))
X_test_tensors_final = torch.reshape(X_test_tensors,
                                     (X_test_tensors.shape[0], x_shape,
                                      X_test_tensors.shape[2]))

C_train_tensors_final = torch.reshape(C_train_tensors,
                                      (C_train_tensors.shape[0], x_shape,
                                       C_train_tensors.shape[2]))
C_test_tensors_final = torch.reshape(C_test_tensors,
                                     (C_test_tensors.shape[0], x_shape,
                                      C_test_tensors.shape[2]))

print("Training Shape:", X_train_tensors_final.shape, C_train_tensors_final.shape, y_train_tensors.shape)
print("Validation Shape:", X_test_tensors_final.shape, C_test_tensors_final.shape, y_test_tensors.shape)
print("Test Shape:", DataX_test.shape, DataC_test.shape, Datay_test.shape)

# %%
def NRMSELoss(yhat,y): #NRMSE
  y = y.reshape(y.shape[0], y.shape[1])
  return (torch.sqrt(torch.mean(torch.square(yhat-y))))/ (torch.max(y) - torch.min(y))

# Root Mean Squared Percentage Error (RMSPE)
def rmspe(y_pred,y_true):
  y_true = y_true.reshape(y_true.shape[0], y_true.shape[1])
  return torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true)))

def MAPELoss(yhat,y):
  y = y.reshape(y.shape[0], y.shape[1])
  return torch.mean(torch.abs(yhat-y)/torch.abs(y))

def SMAPELoss(y_pred, y_true):
  y_true = y_true.reshape(y_true.shape[0], y_true.shape[1])
  loss = 2 * torch.mean(torch.abs(y_true - y_pred) / (torch.max(y_true) + torch.max(y_true)))
  return loss

def MSELoss(y_pred, y_true):
  y_true = y_true.reshape(y_true.shape[0], y_true.shape[1])
  loss = torch.mean((y_pred - y_true)**2)
  return loss

def RMSELoss(y_pred, y_true):
  y_true = y_true.reshape(y_true.shape[0], y_true.shape[1])
  loss = torch.sqrt(torch.mean((y_pred - y_true)**2))
  return loss

# %%

# !pip install tensorboard
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('logs')

# def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=200):
#     lr = init_lr * (0.9**(epoch // lr_decay_epoch))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer

from torch.utils.data import TensorDataset, DataLoader

def training_loop(n_epochs, learning_rate, lr_decay_epoch, network, optimiser, loss_fn, X_train, Condition_train, y_train, X_test, Condition_test, y_test, batch_size):
    loss_valid_show, loss_train_show = [], []

    if torch.cuda.is_available():
        network.cuda()

    best_valid_loss = float('inf')
    best_epoch = -1

    # Split data into batches
    train_loader = DataLoader(TensorDataset(X_train, Condition_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Condition_test, y_test), batch_size=batch_size)

    for epoch in range(n_epochs):
        network.train()
        train_loss = 0.0
        for i, (X_batch, Condition_batch, y_batch) in enumerate(train_loader):
            optimiser.zero_grad()
            outputs = network.forward(X_batch, Condition_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        # Calculate average training loss across batches
        train_loss /= len(train_loader)

        network.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, (X_batch, Condition_batch, y_batch) in enumerate(test_loader):
                outputs = network.forward(X_batch, Condition_batch)
                loss = loss_fn(outputs, y_batch)
                test_loss += loss.item()

        # Calculate average testing loss across batches
        test_loss /= len(test_loader)

        loss_valid_show.append(test_loss)
        loss_train_show.append(train_loss)


        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            best_epoch = epoch
            torch.save({'epoch': epoch, 'state_dict': network.state_dict()}, f'./weights/model_epoch_{epoch}.pth')

        if (epoch) % (50) == 0:
            print("Epoch: %4d, train loss: %1.5f, test loss: %1.5f" % (epoch, train_loss, test_loss))

    # writer.close()
    return loss_train_show, loss_valid_show, best_valid_loss, best_epoch

# %%
%cd D:\projects\Seminar\lingam\Doc\cods\seriesnet_based

import importlib
import Decoder_DARLM
import Encoder_DARLM
import ts_lstm
import ts_gru
import HSAM
import CBAM
import sfm

importlib.reload(CBAM)
importlib.reload(Decoder_DARLM)
importlib.reload(TSASeriesNet)
importlib.reload(Encoder_DARLM)
importlib.reload(ts_lstm)
importlib.reload(ts_gru)
importlib.reload(HSAM)
importlib.reload(sfm)

n_epochs = 2
num_inputs = X_train_tensors_final.shape[1]
dilation_c = 2
kernel_size_EN = 2
kernel_size_DE = 2
hidden_size_lstm =10

num_levels_en = 2
num_levels_de = 2
num_layers_lstm = 3
num_layers_gru = 3
features = X_train_tensors_final.shape[2]
features_c = C_train_tensors_final.shape[2]
output_num = y_train_tensors.shape[1]
lr_decay_epoch = 49
learning_rate = 0.001
weight_decay = 0.001
batch_size = 1000
loss_fn = RMSELoss #SMAPELoss #MAPELoss #NRMSELoss #rmspe #RMSELoss
# ANN model
myModel = TSASeriesNet.ANNmodel(num_inputs, features_c, features, output_num, num_levels_en,num_levels_de, kernel_size_EN, kernel_size_DE, dilation_c, hidden_size_lstm, num_layers_lstm, num_layers_gru ).to(device)
optimiser = torch.optim.Adam(myModel.parameters(),weight_decay=weight_decay)#, lr=learning_rate)#, eps=1e-08)#

loss_train_show ,loss_valid_show , best_valid_loss , best_epoch= training_loop(n_epochs=n_epochs,
                                  learning_rate = learning_rate,
                                  lr_decay_epoch = lr_decay_epoch,
                                  network=myModel,
                                  optimiser=optimiser,
                                  loss_fn=loss_fn,
                                  X_train=X_train_tensors_final,
                                  Condition_train = C_train_tensors_final,
                                  y_train=y_train_tensors,
                                  X_test=X_test_tensors_final,
                                  Condition_test  = C_test_tensors_final,
                                  y_test=y_test_tensors,
                                  batch_size = batch_size)

print(f"best validation loss: {best_valid_loss} in epoch {best_epoch}")

checkpoint = torch.load(f'./weights/model_epoch_{best_epoch}.pth')
myModel.load_state_dict(checkpoint['state_dict'])

# %%


for i in range(num_levels_en):
    weights = pd.DataFrame(myModel.en_darlm.network[i].pointwise_conv.weight.cpu().detach().numpy().reshape(myModel.en_darlm.network[i].pointwise_conv.weight.size(0), -1))
    row_means = np.mean((weights), axis=1)
    row_means_df = pd.DataFrame(row_means)
    row_means_df.index = col
    result_df = np.tanh(row_means_df) #row_means_df.applymap(lambda x: 1 if x > row_means_df[0].mean() else 0 if x == row_means_df[0].mean() else 0)
    # print(result_df)
    result_df.to_csv(f"causal/pointwise_conv_weights_mean{i}.csv")

checkpoint = torch.load(f'./weights/model_epoch_{best_epoch}.pth')
myModel.load_state_dict(checkpoint['state_dict'])

# with torch.no_grad():
#   test_predict = myModel(DataX_test, DataC_test)  # get the last sample
# test_predict = test_predict.detach().cpu().numpy()
# test_predict = MMScaler_test.inverse_transform(test_predict)

# test_target = Datay_test.detach().cpu().numpy()  # last sample again
# %%

with torch.no_grad():
  test_predict = myModel(DataX_test, DataC_test)  # get the last sample
test_predict = test_predict.detach().cpu().numpy()

test_target = Datay_test.detach().cpu().numpy()  # last sample again

def NRMSELoss_test(yhat,y): #NRMSE
  return np.sqrt(np.mean(np.square(yhat-y)))

# NRMSELoss_res = NRMSELoss_test(test_predict[:,0] ,first_day_test_target.reshape(-1) )
NRMSELoss_res = NRMSELoss_test(test_predict ,test_target.reshape(test_target.shape[0],test_target.shape[1]) )
print(NRMSELoss_res)

# %%

checkpoint = torch.load(f'./weights/model_epoch_{best_epoch}.pth')
myModel.load_state_dict(checkpoint['state_dict'])


df_X_ss = SScaler.transform(pd.DataFrame(np.array(Y).reshape(-1, 1))) # old transformers

df_C_ss = condscaler.transform(condtemp)#[f_col])


df_y_mm = MMScaler.transform(pd.DataFrame(np.array(Y).reshape(-1, 1)))
df_y_mm = df_y_mm.squeeze()
# split the sequence
df_X_ss, df_C_ss, df_y_mm = split_sequences(df_X_ss, df_C_ss, df_y_mm, x_shape, y_shape)
# converting to tensors
df_X_ss = Variable(torch.Tensor(df_X_ss))
df_C_ss = Variable(torch.Tensor(df_C_ss))
df_y_mm = Variable(torch.Tensor(df_y_mm))
# reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], x_shape, df_X_ss.shape[2]))
df_C_ss = torch.reshape(df_C_ss, (df_C_ss.shape[0], x_shape, df_C_ss.shape[2]))

predict_loader = DataLoader(TensorDataset(df_X_ss, df_C_ss), batch_size=500)

# make predictions for each batch and concatenate the results
predictions = []
i=0
with torch.no_grad():
  for X_batch, C_batch in predict_loader:
      # print(i)
      # pdb.set_trace()
      # i+=1
      prediction_batch = myModel(X_batch.to(device), C_batch.to(device)).cpu()
      predictions.append(prediction_batch)
predictions = torch.cat(predictions)

# convert predictions and ground truth to numpy arrays
predicted_values = predictions.data.numpy().squeeze()
ground_truth = df_y_mm.data.numpy().squeeze()

# inverse transform the data to get the original scale
predicted_values = MMScaler.inverse_transform(predicted_values)
ground_truth = MMScaler.inverse_transform(ground_truth)

true, preds = [], []
for i in range(len(ground_truth)):
    true.append(ground_truth[i][0])
for i in range(len(predicted_values)):
    preds.append(predicted_values[i][0])


plt.figure(figsize=(100,50)) #plotting
plt.axvline(x=train_test_cutoff -(x_shape), c='r', linestyle='--') # size of the training set

# plot the results
plt.xticks(range(0,total_samples,100))
plt.grid(color='g', linestyle=':', linewidth=0.5)
plt.plot(true, label='Actual Data') # actual plot
plt.plot(preds, label='Predicted Data') # predicted plot
plt.title('Time-Series Prediction')
plt.legend()
# plt.savefig("whole_plot.png", dpi=300)
plt.show()

plt.figure(figsize=(25,10)) #plotting
plt.plot(loss_valid_show, label='ERROR_VALID')
plt.plot(loss_train_show, label='ERROR_TRAIN')
plt.title('Time-Series Prediction ERROR')
plt.grid(color='g', linestyle=':', linewidth=0.5)
plt.xticks(range(0,n_epochs+1,5))
plt.legend()
plt.show()
# print(loss_show)

# %%
plt.figure(figsize=(10,5)) #plotting
with torch.no_grad():
  test_predict = myModel(DataX_test, DataC_test)  # get the last sample
test_predict = test_predict.detach().cpu().numpy()
test_predict = MMScaler_test.inverse_transform(test_predict)
first_day_test_predict = test_predict[:, 0]  # get the first day of the 7-day predictions

test_target = Datay_test.detach().cpu().numpy()  # last sample again
test_target = MMScaler_test.inverse_transform(test_target.flatten().reshape(1, -1))
first_day_test_target = test_target[:, ::y_shape]  # get the corresponding first day in true price

plt.grid(color='g', linestyle='--', linewidth=0.5)
plt.plot(first_day_test_target[0], label="Actual Data")
plt.plot(first_day_test_predict, label="Network Predictions")
plt.legend()
plt.show()

# %%
def reshape_df(df,l=90):
    a = list()
    for i in range(len(df)-l+1):
        seq_x = df[i:l+i]
        a.append(seq_x)
    print(np.array(a).shape)
    return a

start = "2023-6-28"
end="2023-7-05"
try :
    top15_test = yf.download("ADA-USD AVAX-USD BNB-USD BTC-USD DOGE-USD DOT-USD ETH-USD LTC-USD MATIC-USD SHIB-USD SOL-USD TRX-USD USDC-USD USDT-USD XRP-USD" ,start=start, end=end , interval ="1h") #1137
except Exception as e:
    print("error")

# top15_test[('Adj Close',)]
top15_test = top15_test.drop([('High',)],axis=1)
top15_test = top15_test.drop([('Low',)],axis=1)
top15_test = top15_test.drop([('Open',)],axis=1)
top15_test = top15_test.drop([('Adj Close',)],axis=1)

top15_test = top15_test.rename_axis('time_stamp')
top15_test = top15_test.replace(0, np.nan)#.fillna(method='ffill')
top15_test = top15_test.fillna(method='bfill')

top15_close = top15_test[('Close',)]
top15_volume = top15_test[('Volume',)]
new_columns = top15_volume.columns.map(lambda x: x.split('-')[0] + '-volume')
top15_volume = top15_volume.rename(columns=dict(zip(top15_volume.columns, new_columns)))

new_columns = top15_volume.columns.map(lambda x: x.split('-')[0] + '-volume')
top15_volume = top15_volume.rename(columns=dict(zip(top15_volume.columns, new_columns)))


col = list(top15_close.columns)+ list(top15_volume.columns)
top15_crypto = pd.DataFrame(np.hstack((top15_close , top15_volume)))
top15_crypto.columns =col

top15_crypto.index = top15_test.index
top15_crypto = top15_crypto.rename(columns={'BTC-USD': 'price_usd_close'})
top15_crypto = top15_crypto.fillna(method='ffill')

xtest = top15_crypto.price_usd_close[-94:]
condtest = top15_crypto.drop("price_usd_close",axis=1)[-94:]
ytest = top15_crypto.price_usd_close[-94:]

MMScaler_test = MinMaxScaler()
SScaler_test = StandardScaler()
condscaler_test = StandardScaler()

X_trans_test = SScaler_test.fit_transform(xtest.values.reshape(-1, 1))
c_trans_test = condscaler_test.fit_transform(condtest.values)
y_trans_test = MMScaler_test.fit_transform(ytest.values.reshape(-1, 1))

X_trans_test = reshape_df(X_trans_test,90)
c_trans_test = reshape_df(c_trans_test,90)
y_trans_test = reshape_df(y_trans_test,90)

X_trans_test = Variable(torch.Tensor(X_trans_test).to(device))
c_trans_test = Variable(torch.Tensor(c_trans_test).to(device))

best_epoch = 168
checkpoint = torch.load(f'./weights/model_epoch_{best_epoch}.pth')
myModel.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    test_predict = myModel(X_trans_test , c_trans_test)  # get the last sample
test_predict = MMScaler_test.inverse_transform(test_predict.cpu().detach().numpy())
# first_day_test_predict = test_predict[:, 0]  # get the first day of the 7-day predictions

print(test_predict[0][0],test_predict[1][0],test_predict[2][0],test_predict[3][0],test_predict[4][0])
# print("",test_predict[0],"\n 00000.000",test_predict[1])

from datetime import timedelta
for i in range(7):
    t = pd.Timestamp(top15_test.index[-1])
    t = t + timedelta(hours=4+i, minutes=30)
    print("at ",t.strftime('%Y-%m-%d %H:%M:%S')," predicted price: ",test_predict[-1][i])

#2023-07-02 20:00:00 30559.683594

print(test_predict[1][0]- test_predict[0][0] > 0)
for i in range(6):
    print(test_predict[-1][i+1]- test_predict[-1][i] > 0)



# %load_ext tensorboard
# %tensorboard --logdir logs

# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/logs

# !pip install pytorch-model-summary
# import pytorch_model_summary as pms
# pms.summary(myModel, torch.zeros(X.shape[0], 30, X.shape[1]).to(device),torch.zeros(X.shape[0], 30, X.shape[1]).to(device), show_input=True, print_summary=True)

checkpoint = torch.load(f'./weights/model_epoch_{best_epoch}.pth')
myModel.load_state_dict(checkpoint['state_dict'])
# DataX_test, DataC_test, Datay_test

predict = myModel(DataX_test, DataC_test)
predict = predict.detach().cpu().numpy()
predict = MMScaler_test.inverse_transform(predict)
# predict = predict[-100:]
true = np.squeeze(Datay_test)
true = MMScaler_test.inverse_transform(true.cpu())
# true = true[-100:]

# calculate the daily returns based on the predicted and true prices
predict_returns = (predict[:, 1:] - predict[:, :-1]) / predict[:, :-1]
true_returns = (true[:, 1:] - true[:, :-1]) / true[:, :-1]

# calculate the position you would have taken based on your prediction
threshold = 0.00001  # 1% return threshold
predict_position = np.where(predict_returns > threshold, 1, -1)
true_position = np.where(true_returns > threshold, 1, -1)

# define a function to simulate the portfolio based on the predicted or true returns
def simulate_portfolio(position, returns, initial_capital):
    # calculate the daily profit and loss based on the position you took and the daily returns
    transaction_cost = 0.001  # 0.1% per transaction
    position_percentage = 0.2  # 10% of capital
    pnl = position * (initial_capital * position_percentage) * (returns - transaction_cost)

    # calculate the cumulative PnL over the entire period
    cumulative_pnl = np.cumsum(pnl)

    # calculate the final portfolio value
    final_value = initial_capital + cumulative_pnl[-1]

    # calculate the profit percentage
    profit_percentage = ((final_value - initial_capital) / initial_capital) * 100

    return final_value, profit_percentage

# simulate the portfolio based on the predicted and true returns for different initial capital values
for cap in [1000, 2000, 5000, 10000]:
    print(f"\nFor initial capital of {cap}:")
    predict_final_value, predict_profit_percentage = simulate_portfolio(predict_position, predict_returns, cap)
    true_final_value, true_profit_percentage = simulate_portfolio(true_position, true_returns, cap)
    print(f"Predicted final value   : {predict_final_value:.2f} with {predict_profit_percentage:.2f}% profit")
    print(f"True final value        : {true_final_value:.2f} with {true_profit_percentage:.2f}% profit")
    print(f"(Predicted/True)% profit: {predict_final_value*100/true_final_value:.2f}% profit")
    print("-----------------------------------------------------------")

from sklearn.model_selection import ParameterGrid

# Define the hyperparameter grid
param_grid = {
    'kernel_size_EN': [2,3,4],
    'kernel_size_DE': [2,3,4],
    'hidden_size_lstm': [15],
    'num_levels_en': [4,5,6],
    'num_levels_de': [4,5,6],
    'num_layers_lstm': [6],
    'num_layers_gru': [4]
}
loss_fn = NRMSELoss
lr_decay_epoch = 41
learning_rate = 0.01
weight_decay = 0.0001
def NRMSELoss_test(yhat,y): #NRMSE
    return (np.sqrt(np.mean(np.square(yhat-y))))/ (np.max(y) - np.min(y))

def run_experiment(params):
    myModel = TSASeriesNet.ANNmodel(X_train_tensors_final.shape[1], C_train_tensors_final.shape[2], X_train_tensors_final.shape[2],y_train_tensors.shape[1],
                                    params['num_levels_en'], params['num_levels_de'], params['kernel_size_EN'],
                                    params['kernel_size_DE'], 2, params['hidden_size_lstm'],
                                    params['num_layers_lstm'], params['num_layers_gru']).to(device)

    optimiser = torch.optim.Adam(myModel.parameters(),weight_decay=0.0001)#, weight_decay=weight_decay, lr=learning_rate)

    loss_train_show ,loss_valid_show , best_valid_loss , best_epoch= training_loop(n_epochs=n_epochs,
                                  learning_rate = learning_rate,
                                  lr_decay_epoch = lr_decay_epoch,
                                  network=myModel,
                                  optimiser=optimiser,
                                  loss_fn=loss_fn,
                                  X_train=X_train_tensors_final,
                                  Condition_train = C_train_tensors_final,
                                  y_train=y_train_tensors,
                                  X_test=X_test_tensors_final,
                                  Condition_test  = C_test_tensors_final,
                                  y_test=y_test_tensors,
                                  batch_size = batch_size)

    checkpoint = torch.load(f'./weights/model_epoch_{best_epoch}.pth')
    myModel.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        test_predict = myModel(DataX_test, DataC_test)  # get the last sample
    test_predict = test_predict.detach().cpu().numpy()

    test_target = Datay_test.detach().cpu().numpy()  # last sample again



    # NRMSELoss_res = NRMSELoss_test(test_predict[:,0] ,first_day_test_target.reshape(-1) )
    NRMSELoss_res = NRMSELoss_test(test_predict ,test_target.reshape(test_target.shape[0],test_target.shape[1]) )
    print("****TEST*** ",NRMSELoss_res)



    return loss_valid_show[-1]

import itertools

def grid_search(param_grid):
    keys, values = zip(*param_grid.items())
    min_loss = float('inf')
    best_params = None

    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print(f"Running experiment with parameters: {params}")
        loss = run_experiment(params)
        print(f"Validation loss: {loss}")

        if loss < min_loss:
            min_loss = loss
            best_params = params

    return best_params, min_loss

best_params, min_loss = grid_search(param_grid)
print(f"Best parameters: {best_params}")
print(f"Minimum validation loss: {min_loss}")

import backtrader as bt
import pandas as pd

checkpoint = torch.load(f'./weights/model_epoch_{best_epoch}.pth')
myModel.load_state_dict(checkpoint['state_dict'])
# DataX_test, DataC_test, Datay_test

predict = myModel(DataX_test, DataC_test)
predict = predict.detach().cpu().numpy()
predict = pd.DataFrame(MMScaler_test.inverse_transform(predict))
predict.index = data.index[-106:]
predict = predict.rename_axis("datetime")

# predict = predict[-100:]
true = np.squeeze(Datay_test)
true = pd.DataFrame(MMScaler_test.inverse_transform(true.cpu()))
true.index = data.index[-106:]
true = true.rename_axis("datetime")

# Load the predicted data
predicted_data = pd.DataFrame(predict[0])
predicted_data = predicted_data.rename(columns={0: 'predicted'})
# predicted_data.index = data.index[-143:]
# Load the true data
true_data = pd.DataFrame(true[0])
true_data = true_data.rename(columns={0: 'true'})
# true_data.index = data.index[-143:]
# Define the strategy
class NeuralNetStrategy(bt.Strategy):
    params = (
        ('threshold', 0.05),
    )

    def __init__(self):
        self.data_predicted = self.datas[0]
        self.data_true = self.datas[1]

    def next(self):
        if self.data_predicted.close[0] > self.data_true.close[0] * (1 + self.params.threshold):
            self.buy()
        elif self.data_predicted.close[0] < self.data_true.close[0] * (1 - self.params.threshold):
            self.sell()

# Create a cerebro instance
cerebro = bt.Cerebro()

# Add the predicted data
predicted_data_feed = bt.feeds.PandasData(dataname=predicted_data)
cerebro.adddata(predicted_data_feed, name='predicted')

# Add the true data
true_data_feed = bt.feeds.PandasData(dataname=true_data)
cerebro.adddata(true_data_feed, name='true')

# Add the strategy
cerebro.addstrategy(NeuralNetStrategy, threshold=0.05)

# Set the cash and commission
cerebro.broker.setcash(1000.0)
cerebro.broker.setcommission(commission=0.001)

# Run the backtest
cerebro.run()

# Print the final portfolio value
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())