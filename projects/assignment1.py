import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Data ------------------ #
YEARS = np.array([1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2019, 2020, 2021, 2022, 2023, 2024, 2025]).reshape(-1,1)
yMin = np.array([8730, 8781, 9449, 10224, 10575, 11070, 11485, 11845, 11580, 11960, 12565, 13645, 14575, 14610, 14450, 13970, 14490, 16395, 17820, 18160, 24000, 24110, 24820, 25980, 27400, 32820, 33350]).reshape(-1,1)
yMax = np.array([14840, 16535, 18328, 19571, 20295, 20325, 19695, 19435, 19785, 24335, 25010, 25450, 26015, 26795, 26670, 24425, 24350, 25805, 25800, 26070, 38565, 38675, 39035, 39730, 40945, 55720, 56070]).reshape(-1,1)

# ------------------ No Feature Learning and No Dynamic Learning ------------------ #
YEARS_recentered = YEARS.astype(np.float32) - 2000.0    # so it doesn't blow up
X_raw = torch.tensor(YEARS_recentered, dtype=torch.float32) # making tensor
yMin_raw = torch.tensor(yMin, dtype=torch.float32)
yMax_raw = torch.tensor(yMax, dtype=torch.float32)
    
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, dtype=torch.float32)
    
    def forward(self, x):
        return self.linear(x) # want 1d output
    
in_features = 1
out_features = 1
modelMin_raw = LinearRegressionModel(in_features, out_features)
modelMax_raw = LinearRegressionModel(in_features, out_features)

criterion = nn.MSELoss()
optimizerMin = optim.SGD(modelMin_raw.parameters(), lr=1e-7) # sgd optimizer; parameter = parameter - lr * gradient
optimizerMax = optim.SGD(modelMax_raw.parameters(), lr=1e-7) # sgd optimizer; parameter = parameter - lr * gradient

#------ training------#

#   yMin
yMin_losses =[]
num_epochs = 10000
for epoch in range(num_epochs):
    # forward pass
    outputs = modelMin_raw(X_raw)

    # calculate loss
    loss = criterion(outputs, yMin_raw) 
    yMin_losses.append(loss.item())

    # backward pass and optimization
    optimizerMin.zero_grad()
    loss.backward()
    optimizerMin.step()
    if epoch % 1000 == 0:
        print(f'Epoch: [{epoch + 1}/{num_epochs}, loss: {loss.item():.4f}]')

[wMin, bMin] = modelMin_raw.parameters()
#centered space
wMin_c = wMin.item()
bMin_c = bMin.item()
#back to original years
wMin = wMin_c
bMin = bMin_c - 2000.0 * wMin_c

#   yMax
yMax_losses =[]
num_epochs = 10000
for epoch in range(num_epochs):
    # forward pass
    outputs = modelMax_raw(X_raw)

    # calculate loss
    loss = criterion(outputs, yMax_raw) 
    yMax_losses.append(loss.item())

    # backward pass and optimization
    optimizerMax.zero_grad()
    loss.backward()
    optimizerMax.step()

    if epoch % 1000 == 0:
        print(f'Epoch: [{epoch + 1}/{num_epochs}, loss: {loss.item():.4f}]')

[wMax, bMax] = modelMax_raw.parameters()
#centered space
wMax_c = wMax.item()
bMax_c = bMax.item()
#back to original years
wMax = wMax_c
bMax = bMax_c - 2000.0 * wMax_c


#------ predict and plot ------ #
modelMin_raw.eval()
modelMax_raw.eval()
with torch.no_grad():
    yMin_fit_raw = modelMin_raw(X_raw).numpy()
    yMax_fit_raw = modelMax_raw(X_raw).numpy()

plt.scatter(YEARS, yMin, label='Min')
plt.scatter(YEARS, yMax, label='Max')

plt.plot(YEARS, yMin_fit_raw, label='MinFitLine', linewidth = 2)
plt.plot(YEARS, yMax_fit_raw, label='MaxFitLine', linewidth = 2)
# plt.plot(YEARS, (wMin * X_raw + bMin).detach().numpy(), label='MinFitLine', linewidth = 2)
# plt.plot(YEARS, (wMax * X_raw + bMax).detach().numpy(), label='MaxFitLine', linewidth = 2)


plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Pytorch')
plt.show()

# ------------------ Normalized and Feature Scaling and Dynamic Learning ------------------ #
# now we have to turn the data into tensors; common practice is to normalize data
x_mean, x_std = YEARS.mean(), YEARS.std()
X_normalized = (YEARS - x_mean) / x_std
X_tensor = torch.tensor(X_normalized, dtype=torch.float32) # making tensor
# print(X_tensor.shape)

yMin_mean, yMin_std = yMin.mean(), yMin.std()
y_normalized = (yMin - yMin_mean) / yMin_std
yMin_tensor = torch.tensor(y_normalized, dtype=torch.float32)
# print(yMin_tensor.shape)

yMax_mean, yMax_std = yMax.mean(), yMax.std()
y_normalized = (yMax - yMax_mean) / yMax_std
yMax_tensor = torch.tensor(y_normalized, dtype=torch.float32)
# print(yMax_tensor.shape)

# ------ model ------ #
class LinearRegressionModelFS(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x) # want 1d output

in_features = 1
out_features = 1
modelMin = LinearRegressionModelFS(in_features, out_features)
modelMax = LinearRegressionModelFS(in_features, out_features)

criterion = nn.MSELoss()
optimizerMin = optim.SGD(modelMin.parameters(), lr=1e-3) # sgd optimizer; parameter = parameter - lr * gradient
optimizerMax = optim.SGD(modelMax.parameters(), lr=1e-3) # sgd optimizer; parameter = parameter - lr * gradient

#------ training------#

#   yMin
yMinFS_losses =[]
num_epochs = 10000
for epoch in range(num_epochs):
    # forward pass
    outputs = modelMin(X_tensor)

    # calculate loss
    loss = criterion(outputs, yMin_tensor) 
    yMinFS_losses.append(loss.item())

    # backward pass and optimization
    optimizerMin.zero_grad()
    loss.backward()
    optimizerMin.step()
    if epoch % 1000 == 0:
        print(f'Epoch: [{epoch + 1}/{num_epochs}, loss: {loss.item():.4f}]')

[wMinFS, bMinFS] = modelMin.parameters()
#centered space
wMinFS_c = wMinFS.item()
bMinFS_c = bMinFS.item()
#back to original years
wMin = wMin_c
bMin = bMin_c - 2000.0 * wMin_c

#   yMax
yMaxFS_losses =[]
num_epochs = 10000
for epoch in range(num_epochs):
    # forward pass
    outputs = modelMax(X_tensor)

    # calculate loss
    loss = criterion(outputs, yMax_tensor) 
    yMaxFS_losses.append(loss.item())

    # backward pass and optimization
    optimizerMax.zero_grad()
    loss.backward()
    optimizerMax.step()

    if epoch % 1000 == 0:
        print(f'Epoch: [{epoch + 1}/{num_epochs}, loss: {loss.item():.4f}]')

[wMaxFS, bMaxFS] = modelMax.parameters()
#centered space
wMaxSS_c = wMaxFS.item()
bMaxFS_c = bMaxFS.item()


#------ predict and plot ------ #
predict_years = np.arange(2012,2019).reshape(-1,1)

years_norm = (predict_years - x_mean) / x_std
years_norm_tensor = torch.tensor(years_norm, dtype=torch.float32)

modelMin.eval()
modelMax.eval()
with torch.no_grad():
    yMin_fit_fs = (modelMin(X_tensor).cpu().numpy() * yMin_std) + yMin_mean
    yMax_fit_fs = (modelMax(X_tensor).cpu().numpy() * yMax_std) + yMax_mean

    yMin_norm_preds = modelMin(years_norm_tensor).cpu().numpy()   # shape (7,1)
    yMax_norm_preds = modelMax(years_norm_tensor).cpu().numpy()


# yMin_preds = (yMin_fit_fs.numpy() * yMin_std) + yMin_mean
# yMax_preds = (yMax_fit_fs.numpy() * yMax_std) + yMax_mean

yMin_preds = (yMin_norm_preds * yMin_std) + yMin_mean
yMax_preds = (yMax_norm_preds * yMax_std) + yMax_mean

plt.scatter(YEARS, yMin, label='Min')
plt.scatter(YEARS, yMax, label='Max')

plt.plot(YEARS, yMin_fit_fs, label='MinFitLine', linewidth = 2)
plt.plot(YEARS, yMax_fit_fs, label='MaxFitLine', linewidth = 2)

plt.scatter(predict_years, yMin_preds, marker='^', label='Min Pred (2012–2018)')
plt.scatter(predict_years, yMax_preds, marker='v', label='Max Pred (2012–2018)')

plt.legend()
plt.xlabel('Years')
plt.ylabel('Prices')
plt.title('Pytorch')
plt.show()


# ------------------ Losses ------------------ #
plt.plot(range(num_epochs), yMin_losses, label="yMin raw")
plt.plot(range(num_epochs), yMax_losses, label="yMax raw")
plt.plot(range(num_epochs), yMinFS_losses, label="yMin FS")
plt.plot(range(num_epochs), yMaxFS_losses, label="yMax FS")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch v losses')
plt.legend()
plt.show()
