import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Data ------------------ #
YEARS = np.array([1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
YEARS = YEARS.reshape(-1,1)
yMin = np.array([8730, 8781, 9449, 10224, 10575, 11070, 11485, 11845, 11580, 11960, 12565, 13645, 14575, 14610, 14450, 13970, 14490, 16395, 17820, 18160, 24000, 24110, 24820, 25980, 27400, 32820, 33350]).reshape(-1,1)
yMax = np.array([14840, 16535, 18328, 19571, 20295, 20325, 19695, 19435, 19785, 24335, 25010, 25450, 26015, 26795, 26670, 24425, 24350, 25805, 25800, 26070, 38565, 38675, 39035, 39730, 40945, 55720, 56070]).reshape(-1,1)

#------ Model ------#
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, dtype=torch.float32)
    
    def forward(self, x):
        return self.linear(x) # want 1d output

# ------------------ No Feature Learning and No Dynamic Learning ------------------ #
YEARS_recentered = YEARS.astype(np.float32) - 2000.0 
# YEARS_recentered = YEARS.astype(np.float32)
X_raw = torch.tensor(YEARS_recentered, dtype=torch.float32) # making tensor
yMin_raw = torch.tensor(yMin, dtype=torch.float32)
yMax_raw = torch.tensor(yMax, dtype=torch.float32)
    
#------ training------#
def train_raw(model, X, y, lr=1e-3, epochs=10000):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr) # replace with something else
    losses = []
    for e in range(epochs):
        pred = model(X) 
        loss = criterion(pred, y) 
        optimizer.zero_grad() # cant use this
        loss.backward() # or this
        optimizer.step() # or this 
        losses.append(loss.item())
        if e % 10 == 0:
            print(f"Epoch [{e+1}/{epochs}], loss: {loss.item():.4f}")
    return losses

in_features = 1
out_features = 1
modelMin_raw = LinearRegressionModel(in_features, out_features)
modelMax_raw = LinearRegressionModel(in_features, out_features)

#   yMin
lr_raw = 1e-9
yMin_raw_losses = train_raw(modelMin_raw, X_raw, yMin_raw, lr_raw, epochs=100)

[wMin, bMin] = modelMin_raw.parameters()
#centered space
wMin_c = wMin.item()
bMin_c = bMin.item()
#back to original years
wMin = wMin_c
bMin = bMin_c - 2000.0 * wMin_c

#   yMax
yMax_losses =[]
yMax_raw_losses = train_raw(modelMax_raw, X_raw, yMax_raw, lr_raw, epochs=100)

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

plt.legend()
plt.xlabel('Years')
plt.ylabel('Prices')
plt.title('Raw Linear Regression')
plt.show()

#------ Loss ------#
plt.plot(range(100), yMin_raw_losses, label="yMin raw")
plt.plot(range(100), yMax_raw_losses, label="yMax raw")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch v losses')
plt.legend()
plt.show()

# ------------------ Feature Scaling and Dynamic Learning ------------------ #
# now we have to turn the data into tensors; common practice is to normalize data
x_mean, x_std = YEARS.mean(), YEARS.std()
X_normalized = (YEARS - x_mean) / x_std
X_tensor = torch.tensor(X_normalized, dtype=torch.float32) # making tensor

yMin_mean, yMin_std = yMin.mean(), yMin.std()
y_normalized = (yMin - yMin_mean) / yMin_std
yMin_tensor = torch.tensor(y_normalized, dtype=torch.float32)

yMax_mean, yMax_std = yMax.mean(), yMax.std()
y_normalized = (yMax - yMax_mean) / yMax_std
yMax_tensor = torch.tensor(y_normalized, dtype=torch.float32)

#------ training------#
def train_DL(model, X, y, lr=1e-1, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # halves LR every 1000 epochs (tweak step_size/gamma)

    losses = []
    for e in range(epochs):
        pred = model(X) 
        loss = criterion(pred, y) 

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if e % 10 == 0:
            curr_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{e+1}/{epochs}], loss: {loss.item():.4f}, currrent lr: {curr_lr:.6g}")
    return losses

in_features = 1
out_features = 1
modelMin_FSDL = LinearRegressionModel(in_features, out_features)
modelMax_FSDL = LinearRegressionModel(in_features, out_features)

#   yMin
lr_FSDL = 1e-1
yMinFSDL_losses = train_DL(modelMin_FSDL, X_tensor, yMin_tensor, lr_FSDL, epochs=100)

[wMinFS, bMinFS] = modelMin_FSDL.parameters()
#centered space
wMinFS_c = wMinFS.item()
bMinFS_c = bMinFS.item()
#back to original years
wMin = wMin_c
bMin = bMin_c - 2000.0 * wMin_c

#   yMax
yMaxFSDL_losses = train_DL(modelMax_FSDL, X_tensor, yMax_tensor, lr_FSDL, epochs=100)

[wMaxFS, bMaxFS] = modelMax_FSDL.parameters()
#centered space
wMaxSS_c = wMaxFS.item()
bMaxFS_c = bMaxFS.item()

#------ predict and plot ------ #
predict_years = np.arange(2012,2019).reshape(-1,1)

years_norm = (predict_years - x_mean) / x_std
years_norm_tensor = torch.tensor(years_norm, dtype=torch.float32)

modelMin_FSDL.eval()
modelMax_FSDL.eval()
with torch.no_grad():
    yMin_fit_fs = (modelMin_FSDL(X_tensor).cpu().numpy() * yMin_std) + yMin_mean
    yMax_fit_fs = (modelMax_FSDL(X_tensor).cpu().numpy() * yMax_std) + yMax_mean

    yMin_norm_preds = modelMin_FSDL(years_norm_tensor).cpu().numpy()
    yMax_norm_preds = modelMax_FSDL(years_norm_tensor).cpu().numpy()

yMin_preds = (yMin_norm_preds * yMin_std) + yMin_mean
yMax_preds = (yMax_norm_preds * yMax_std) + yMax_mean

plt.scatter(YEARS, yMin, label='Min')
plt.scatter(YEARS, yMax, label='Max')

plt.plot(YEARS, yMin_fit_fs, label='MinFitLine', linewidth = 2)
plt.plot(YEARS, yMax_fit_fs, label='MaxFitLine', linewidth = 2)

plt.scatter(predict_years, yMin_preds, marker='v', label='Min Pred (2012–2018)')
plt.scatter(predict_years, yMax_preds, marker='v', label='Max Pred (2012–2018)')

plt.legend()
plt.xlabel('Years')
plt.ylabel('Prices')
plt.title('FSDL Linear Regression') 
plt.show()

#------ Losses ------#
plt.plot(range(100), yMinFSDL_losses, label="yMin FSDL")
plt.plot(range(100), yMaxFSDL_losses, label="yMax FSDL")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch v losses')
plt.legend()
plt.show()