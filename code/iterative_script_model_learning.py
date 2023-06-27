# %% Package imports
import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToDevice
import xarray as xr
import matplotlib.pyplot as plt

# %% Class imports
from load_pygeo_data import PyGeoDataLoader as PyGeoDataLoader
from plot_result_functions import ResultSupportFunctions as support_f
'''
Settings begin
'''
# %% Load model
from Model_MLPs import MLP_main as my_MLP # MLP_main, MLP_main_incl_pmf_wind, MLP_main_incl_pdf_wind
# %% Experiment _idx
_idx = 0
# %% Model and loss settings
num_epoch = 200
print_epochs_status = 10
# During which Epoch should an extensive results XArray be calculated. Note: Extremely slow, set to num_epoch or higher to disable
calc_detailed_results = np.array((num_epoch+1)) #np.floor(num_epoch/2)

hidden_channels_amount = 8
hidden_channels_size = 64
wind_channel = None
dropout = 0
batch_norms = [False]#[True, False, False]
acts = ['lrelu']#['relu', 'relu', 'lrelu']
#act = [None, None, None, 'lrelu', 'lrelu', 'lrelu', 'relu', 'relu', 'relu']
act_firsts = [False]#[False, False, False]
layer_biass = [True]#[True, True, True]

loss_function_edge = ['l1_loss', 'mse_loss', 'mape']
loss_function_edge_train = loss_function_edge[1]
loss_function_edge_test = loss_function_edge[1]
# Set data paths, wind_rose reduction methods, and filter methods/thresholds for single OR multi wind data
wind_rose_reduction_methods = ['no_reduction', 'cardinal_reduction', 'prominent_reduction']
filter_methods = ['no_filter', 'random_filter', 'cartesian_filter', 'normalized_filter', 'pwl_filter', 'random_filter']
filter_thresholds = [0, 0.8, 1.3, 0.5, 1e-3, 0.2]

wind_rose_reduction_methods = wind_rose_reduction_methods[0:1]
filter_methods = filter_methods[5:]
filter_thresholds = filter_thresholds[5:]

doc_path = r'C:\Studies\WFLO-GNN\documents'
data_path = os.path.join(doc_path, 'data', 'data_single_pmf_wd')
#data_path = os.path.join(doc_path, 'data', 'data_multi_wd')
result_path = os.path.join(data_path, 'results_detailed_time_2')
'''
Settings end
'''
raw_dir_path = os.path.join(data_path, 'raw')
pam_dir_path = os.path.join(data_path, 'pam')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

for batch_norm, act, act_first, layer_bias in zip(batch_norms, acts, act_firsts, layer_biass):
    for wind_rose_reduction_method in wind_rose_reduction_methods:
        for filter_method, filter_threshold in zip(filter_methods, filter_thresholds):
            saving_dir = f'{_idx}_{wind_rose_reduction_method}-{filter_method}-{filter_threshold}-layer_{hidden_channels_amount}_{hidden_channels_size}-batchn_{batch_norm}-act_{act}-actf_{act_first}-lbias_{layer_bias}'
            print(f'Beginning model with: {saving_dir}')
            results_dir_path = os.path.join(result_path, f'{saving_dir}')

            # %% PyGeo data setup
            dataset = PyGeoDataLoader(
                root=data_path,
                wind_rose_reduction_method=wind_rose_reduction_method,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
                transform=Compose([
                    ToDevice(device)
                ])
            )

            dataset = dataset.shuffle()
            split_point = int(np.floor(dataset.len()*80/100))
            train_dataset = dataset[0:split_point]
            test_dataset = dataset[split_point:dataset.len()]

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) # Shuffle=False for the detailed data results, else it cannot be linked to the same farm
            train_loader.num_workers = 0 # This could be increased for multithreading but produced a lot of errors, due to a large list of possible issues. (e.g. using Windows Python, Visual Studio not supporting mthreading, CUDA installation/options, CMAKE_GENERATOR,...). Decided not to waste more time on this.
            train_loader.pin_memory = False # Torch documentation mentioned setting this to True for faster GPU computation, but forums mention the error is due to the data already being on GPU. 
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            test_loader.num_workers = 0 # This could be increased for multithreading but produced a lot of errors, due to a large list of possible issues. (e.g. using Windows Python, Visual Studio not supporting mthreading, CUDA installation/options, CMAKE_GENERATOR,...). Decided not to waste more time on this.
            test_loader.pin_memory = False # Torch documentation mentioned setting this to True for faster GPU computation, but forums mention the error is due to the data already being on GPU. 

            # %% PyGeo model setup
            model = my_MLP(
                in_channel_size = dataset.num_edge_features,
                out_channel_size = 1,
                hidden_channels_size = hidden_channels_size,
                hidden_channels_amount = hidden_channels_amount,
                #wind_channel = wind_channel,
                dropout = dropout,
                batch_norm = batch_norm,
                act_f = act,
                act_first = act_first,
                layer_bias = layer_bias
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters()
            )

            # %% Training/Testing loop
            # Collection arrays of losses over the epochs for the training and testing of edge level MSE
            train_loss_edge_epochs, test_loss_edge_epochs = np.zeros(shape=(num_epoch, len(train_loader))), np.zeros(shape=(num_epoch, len(test_loader)))
            # Training time for training and testing functions
            train_f_time_epochs, test_f_time_train_epochs, test_f_time_test_epochs = np.zeros(shape=(num_epoch, len(train_loader))), np.zeros(shape=(num_epoch, len(train_loader))), np.zeros(shape=(num_epoch, len(test_loader)))
            test_loss_detailed = xr.DataArray()

            for epoch in range(num_epoch):

                # Training the model
                train_f_time_epochs[epoch] = support_f.train_model(model, optimizer, train_loader, loss_function_edge_train)

                # Testing the model results
                train_loss_edge_epochs[epoch], test_f_time_train_epochs[epoch] = support_f.test_model(model, train_loader, loss_function_edge_test)
                test_loss_edge_epochs[epoch], test_f_time_test_epochs[epoch] = support_f.test_model(model, test_loader, loss_function_edge_test)
                
                if np.any(calc_detailed_results == epoch):
                    test_loss_detailed = support_f.test_model_detailed_multiple_farms(model, test_loader, epoch, test_loss_detailed)
                if (((epoch % (num_epoch / print_epochs_status)) == 0) or (epoch == num_epoch-1)):
                    print(f'Finished epoch {epoch}, current edge loss: {test_loss_edge_epochs[epoch].mean()}')# and farm loss: {test_loss_farm_epochs[epoch].mean()}')

            try:
                torch.save(model, os.path.join(results_dir_path, 'model'))
            except:
                os.makedirs(results_dir_path)
                torch.save(model, os.path.join(results_dir_path, 'model'))

            # %% Results - Transform, show, and save
            try:
                xr.DataArray.to_netcdf(test_loss_detailed, path=os.path.join(results_dir_path, 'test_loss_detailed_all.nc'))
            except:
                os.makedirs(results_dir_path)
                xr.DataArray.to_netcdf(test_loss_detailed, path=os.path.join(results_dir_path, 'test_loss_detailed_all.nc'))

            train_f_time_epochs_flat = train_f_time_epochs.flatten()
            test_f_time_train_epochs_flat = test_f_time_train_epochs.flatten()
            test_f_time_test_epochs_flat = test_f_time_test_epochs.flatten()

            train_loss_edge_epochs_mean = train_loss_edge_epochs.mean(axis=1)
            test_loss_edge_epochs_mean = test_loss_edge_epochs.mean(axis=1)

            print(f'Training time total: {train_f_time_epochs_flat.sum()}, mean: {train_f_time_epochs_flat.mean()}, stddev: {train_f_time_epochs_flat.std()}, min: {train_f_time_epochs_flat.min()}, max: {train_f_time_epochs_flat.max()}\nTesting time total: {test_f_time_test_epochs_flat.sum()}, mean: {test_f_time_test_epochs_flat.mean()}, stddev: {test_f_time_test_epochs_flat.std()}, min: {test_f_time_test_epochs_flat.min()}, max: {test_f_time_test_epochs_flat.max()}')
            #training_times = np.array((train_f_time_epochs_flat.sum(), train_f_time_epochs_flat.mean(), train_f_time_epochs_flat.std(), train_f_time_epochs_flat.min(), train_f_time_epochs_flat.max(), test_f_time_test_epochs_flat.sum(), test_f_time_test_epochs_flat.mean(), test_f_time_test_epochs_flat.std(), test_f_time_test_epochs_flat.min(), test_f_time_test_epochs_flat.max()))
            #training_times = np.array((train_f_time_epochs_flat, test_f_time_test_epochs_flat))

            try:
                np.savetxt(os.path.join(results_dir_path, 'Training_times.txt'), train_f_time_epochs_flat)
                np.savetxt(os.path.join(results_dir_path, 'Testing_times.txt'), test_f_time_test_epochs_flat)
                np.savetxt(os.path.join(results_dir_path, 'Losses_epochs.txt'), np.array((train_loss_edge_epochs_mean, test_loss_edge_epochs_mean)))
            except:
                os.makedirs(results_dir_path)
                np.savetxt(os.path.join(results_dir_path, 'Training_times.txt'), train_f_time_epochs_flat)
                np.savetxt(os.path.join(results_dir_path, 'Testing_times.txt'), test_f_time_test_epochs_flat)
                np.savetxt(os.path.join(results_dir_path, 'Losses_epochs.txt'), np.array((train_loss_edge_epochs_mean, test_loss_edge_epochs_mean)))

            support_f.plot_y_y(
                'Edge level losses',
                train_loss_edge_epochs_mean,
                test_loss_edge_epochs_mean,
                'Training',
                'Testing'
            )

            try:
                plt.savefig(os.path.join(results_dir_path, 'Edge Loss.png'), format='png')
            except:
                os.makedirs(results_dir_path)
                plt.savefig(os.path.join(results_dir_path, 'Edge Loss.png'), format='png')
            print(f'Finished model with: {saving_dir}')
        _idx += 1