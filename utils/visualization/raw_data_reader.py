import netCDF4
import numpy as np
from keras.models import load_model
from utils.evaluation.reconstruction_models_evaluator import evaluate_single_case
from matplotlib import pyplot as plt
import cmocean


def myround(x, base=5):
    return base * round(x/base)


plt.rcParams.update({'font.size': 14})
model_path = r'C:\Users\Al-Najar\Desktop\SGD_fulldataset_msle-weights-improvement-140-0.01.hdf5'
model = load_model(model_path)

filename = 'output_B2'
base_path = r'C:\Users\Al-Najar\Downloads\alloutput\\' + filename
nc_bathy = base_path + r'\dep.nc'
nc_waves = base_path + r'\eta.nc'

bathy = netCDF4.Dataset(nc_bathy)['depth']
bathy = np.array(bathy).astype(float)

waves_sim = netCDF4.Dataset(nc_waves)['eta']
bst = waves_sim[52:56]

burst = np.rollaxis(bst, 0, 3)
burst = (burst - np.min(burst[:, :100, :])) / np.max(burst[:, :100, :])
burst = np.expand_dims(burst, axis=0)
res = model.predict(burst)[0, :, :, 0] * 10

eval_res = evaluate_single_case(model_path=model_path, burst=burst, bathy=bathy)
print('rmse: ' + str(eval_res['rmse']))
print('rmse_middle: ' + str(eval_res['rmse_middle']))
print('rmse_between_0_and_40: ' + str(eval_res['rmse_between_0_and_40']))
print('0m index: ' + str(eval_res['index_0m']))
print('40m index: ' + str(eval_res['index_40m']))

vmin = np.min(-res)
if np.min(-bathy) < vmin:
    vmin = np.min(-bathy)
print(vmin)

vmax = np.max(-res)
if np.max(-bathy) > vmax:
    vmax = np.max(-bathy)
print(vmax)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
cmap = 'deep_r'
im1 = ax1.imshow(-res, cmap=cmocean.cm.deep_r, vmin=vmin, vmax=vmax, extent=[0, 2000, 2000, 0])
ax1.set_title('Deep learning model prediction\nRMSE = ' + '{:.2f}'.format(eval_res['rmse_middle'], 2) + ' meters')
im2 = ax2.imshow(-bathy, cmap=cmocean.cm.deep_r, vmin=vmin, vmax=vmax, extent=[0, 2000, 2000, 0])
ax2.set_title('Real')
fig.text(0.5, 0.04, 'Cross-shore coordinates [m]', ha='center')
fig.text(0.07, 0.5, 'Alongshore coordinates [m]', va='center', rotation='vertical')
plt.savefig('../plots/' + filename + '_bathys_without_cbar.png')

plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.savefig('../plots/' + filename + '_bathys_with_cbar.png')

avg_slope_real = np.average(bathy, axis=0)
avg_slope_predicted = np.average(res, axis=0)

fig1 = plt.figure(figsize=(15, 7))

ax3 = fig1.add_subplot(121)
ax3.plot(-avg_slope_real, label='real', color='green')
ax3.plot(-avg_slope_predicted, label='prediction', color='blue', linestyle='--')
ax3.set_xticklabels(np.arange(-250, 2001, 250))
ax3.legend(['Input', 'Estimation'])
ax3.set_title('Deep learning model prediction vs synthetic average profiles')

index_0m = eval_res['index_0m']
index_40m = eval_res['index_40m']
dif = index_0m - index_40m
r = np.linspace(index_40m, index_0m + 10, dif) * 10
print(r.shape)
print(avg_slope_real.shape)
# exit()
ax2 = fig1.add_subplot(122)
ax2.plot(r, -avg_slope_real[eval_res['index_40m']:eval_res['index_0m']], label='real', color='green')
ax2.plot(r, -avg_slope_predicted[eval_res['index_40m']:eval_res['index_0m']], label='prediction', color='blue', linestyle='--')
ax2.legend(['Input', 'Estimation'])
ax2.set_title('Deep learning model prediction vs synthetic average profiles\nDepth range: 0m to -40m\nPrediction RMSE = ' +
          '{:.2f}'.format(eval_res['rmse_between_0_and_40'], 2) + ' meters')

fig1.text(0.5, 0.04, 'Cross shore coordinates [m]', ha='center')
fig1.text(0.07, 0.5, 'Depth [m]', va='center', rotation='vertical')
plt.savefig('../plots/' + filename + '_average_profiles.png')

# with open('../plots/' + filename + '_res.json', 'w') as json_file:
#     json.dump(eval_res, json_file)

x = np.arange(-9, 200-9, 1) # cross-shore
y = np.arange(0, 200, 1)
X, Y = np.meshgrid(x, y)

# fig5 = plt.figure(figsize=(15, 10), dpi=100)
# ax5 = fig5.gca(projection='3d')
# ax5.set_title('Real')
# im5 = ax5.plot_surface(X, Y, -bathy[:, :], cmap='ocean_r', linewidth=0.5)
# ax5.plot_surface(X, Y, burst[0, :, :, 3], cmap='Blues', linewidth=0.5)
# ax5.view_init(elev=18, azim=92)
# ax5.grid(b=False)
# plt.colorbar(im5, ax=ax5, shrink=0.5)
#
# fig6 = plt.figure(figsize=(15, 10), dpi=100)
# ax6 = fig6.gca(projection='3d')
# ax6.set_title('Prediction')
# im6 = ax6.plot_surface(X, Y, -res[:, :], cmap='ocean_r', linewidth=0.5)
# ax6.plot_surface(X, Y, burst[0, :, :, 3], cmap='Blues', linewidth=0.5)
# ax6.view_init(elev=18, azim=92)
# ax6.grid(b=False)
# plt.colorbar(im6, ax=ax6, shrink=0.5)

# plt.show()
