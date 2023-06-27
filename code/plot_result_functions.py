import torch.nn.functional as F
import torch
import numpy as np
import timeit
from ruamel.yaml import YAML as _yaml
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import math
from torchmetrics.functional import mean_absolute_percentage_error

class ResultSupportFunctions():
    '''
    Class which contains a variety of functions to plot and get results for the WFLO-GNN model and data.
    '''
    def yaml_load(f):
        '''Returns a YAML file from the given path f
        '''
        return _yaml(typ='safe').load(f)
 
    def train_model(model, optimizer, loader, loss_function = 'l1_loss'):
        '''Trains the given model using the given optimizer and loader. 
        Input: model, optimizer, loader, loss_function
        Output: Training time for each batch
        '''

        model.train()

        train_times_per_batch = np.zeros(shape=(len(loader)))

        try: 
            f_loss_function = getattr(F, loss_function)
        except: 
            print(f'WARNING: The loss function parameter: {loss_function} was not recognized, defaulting to l1_loss.')
            f_loss_function = getattr(F, 'mse_loss')

        for i, data in enumerate(loader):
            startTime = timeit.default_timer()

            optimizer.zero_grad()
            out = model(data)
            loss = f_loss_function(out, data.y)
            loss.backward()
            optimizer.step()

            train_times_per_batch[i] = (timeit.default_timer() - startTime)
        return train_times_per_batch

    def test_model(model, loader, loss_function_edge = 'mape'):#, loss_function_batch = 'l1_loss', loss_as_percentage = False):
        '''Tests the given model with the loader for each edge and batch. 
        Input: model, loader, loss_function_edge, loss_function_batch
        Output: Losses on edge level and batch level, and test time per batch
        '''
        model.eval()

        test_time_per_batch = np.zeros(shape=(len(loader)))
        losses_edge_level = np.zeros(shape=(len(loader)))

        try: 
            f_loss_function_edge = getattr(F, loss_function_edge)
        except: 
            print(f'WARNING: The loss function parameter: {loss_function_edge} was not recognized, defaulting to l1_loss.')
            f_loss_function_edge = getattr(F, 'mse_loss')
        
        for i, data in enumerate(loader):
            startTime = timeit.default_timer()
            
            # Test edge
            out_edges = model(data)
            losses_edge_level[i] = f_loss_function_edge(out_edges, data.y).detach().item()

            test_time_per_batch[i] = (timeit.default_timer() - startTime)
        return losses_edge_level, test_time_per_batch #losses_batch_level,

    def test_model_detailed(model, batch, epoch, previousEpochXArrays):
        '''Tests the given model on a single layout/batch and returns more detailed results. 
        Input: model, one layout/batch, which epoch this is, an XArray DataArray.
        Output: An XArray DataArray containing detailed results for a single layout.

        The input DataArray can either be empty, or a DataArray returned from this same function. It will automatically combine the two. When first calling this function, use an empty DataArray
        
        For the purposes of evaluating the model in detail
        '''
        raise DeprecationWarning("Use test_model_detailed_multiple_farms() instead.")
        model.eval()
        
        source_values, target_values, edge_feat_values, epoch_value = np.arange(0, len(batch.x), 1), np.arange(0, len(batch.x), 1), np.array(("pwl_y", "pwl_pred", "loss_mae", "loss_mse", "loss_prct", "c_dist", "n_dist")), np.array([epoch])
        results_array = np.full((source_values.size, target_values.size, edge_feat_values.size, epoch_value.size), np.NAN)
        returnXArray = xr.DataArray(results_array, dims=("source", "target", "edge_feature", "epoch"), coords={"source": source_values, "target": target_values, "edge_feature": edge_feat_values, "epoch": epoch_value})
        
        pwl_pred = model(batch.edge_attr)
        losses_detailed_mae = F.l1_loss(pwl_pred, batch.y, reduction='none')
        losses_detailed_mse = F.mse_loss(pwl_pred, batch.y, reduction='none')
        losses_detailed_prct = ((losses_detailed_mae * 100) / batch.y)
        
        for edge in range(len(batch.y)):
            source = batch.edge_index[0][edge].detach().item()
            target = batch.edge_index[1][edge].detach().item()
            returnXArray.loc[dict(source = source, target = target, edge_feature = 'pwl_y')] = batch.y[edge].detach().item()
            returnXArray.loc[dict(source = source, target = target, edge_feature = 'pwl_pred')] = pwl_pred[edge].detach().item()
            returnXArray.loc[dict(source = source, target = target, edge_feature = 'loss_mae')] = losses_detailed_mae[edge].detach().item()
            returnXArray.loc[dict(source = source, target = target, edge_feature = 'loss_mse')] = losses_detailed_mse[edge].detach().item()
            returnXArray.loc[dict(source = source, target = target, edge_feature = 'loss_prct')] = losses_detailed_prct[edge].detach().item()
            returnXArray.loc[dict(source = source, target = target, edge_feature = 'c_dist')] = math.sqrt((batch.edge_attr[edge][0])**2 + (batch.edge_attr[edge][1])**2)
            returnXArray.loc[dict(source = source, target = target, edge_feature = 'n_dist')] = math.sqrt((batch.edge_attr[edge][0] / batch.site_radius)**2 + (batch.edge_attr[edge][1] / batch.site_radius)**2)
        return returnXArray.combine_first(previousEpochXArrays)

    def test_model_detailed_multiple_farms(model, batches, epoch, previousEpochXArrays):
        '''Tests the given model on multiple batches and returns more detailed results. 
        Input: model, multiple farms/batches, which epoch this is, an XArray DataArray.
        Output: An XArray DataArray containing detailed results for each layout.

        The input DataArray can either be empty, or a DataArray returned from this same function. It will automatically combine the two. When first calling this function, use an empty DataArray
        
        For the purposes of evaluating the model in detail
        '''
        model.eval()

        edge_feat_values = np.array(("pwl_y", "pwl_pred", "loss_mae", "loss_mse", "loss_prct", "c_dist", "n_dist"))
        epoch_value = np.array([epoch])
        
        for i, batch in enumerate(batches):
            farm_value = np.array([i])
            source_values, target_values = np.arange(0, len(batch.x), 1), np.arange(0, len(batch.x), 1)
            results_array = np.full((farm_value.size, source_values.size, target_values.size, edge_feat_values.size, epoch_value.size), np.NAN)
            returnXArray = xr.DataArray(results_array,
                dims=("farm_id", "source", "target", "edge_feature", "epoch"), 
                coords={"farm_id": farm_value, "source": source_values, "target": target_values, "edge_feature": edge_feat_values, "epoch": epoch_value}
            )

            pwl_pred = model(batch)
            losses_detailed_mae = F.l1_loss(pwl_pred, batch.y, reduction='none')
            losses_detailed_mse = F.mse_loss(pwl_pred, batch.y, reduction='none')
            abs_diff = torch.abs((1 - pwl_pred) - (1 - batch.y))
            losses_detailed_prct = abs_diff / torch.clamp(torch.abs((1 - batch.y)), min=1.17e-06)
            #abs_diff = torch.abs(pwl_pred - batch.y)
            #losses_detailed_prct = abs_diff / torch.clamp(torch.abs(batch.y), min=1.17e-06)
            
            for edge in range(len(batch.y)):
                source = batch.edge_index[0][edge].detach().item()
                target = batch.edge_index[1][edge].detach().item()
                returnXArray.loc[dict(source = source, target = target, edge_feature = 'pwl_y')] = batch.y[edge].detach().item()
                returnXArray.loc[dict(source = source, target = target, edge_feature = 'pwl_pred')] = pwl_pred[edge].detach().item()
                returnXArray.loc[dict(source = source, target = target, edge_feature = 'loss_mae')] = losses_detailed_mae[edge].detach().item()
                returnXArray.loc[dict(source = source, target = target, edge_feature = 'loss_mse')] = losses_detailed_mse[edge].detach().item()
                returnXArray.loc[dict(source = source, target = target, edge_feature = 'loss_prct')] = losses_detailed_prct[edge].detach().item()
                returnXArray.loc[dict(source = source, target = target, edge_feature = 'c_dist')] = math.sqrt((batch.edge_attr[edge][0])**2 + (batch.edge_attr[edge][1])**2)
                returnXArray.loc[dict(source = source, target = target, edge_feature = 'n_dist')] = math.sqrt((batch.edge_attr[edge][0] / batch.site_radius)**2 + (batch.edge_attr[edge][1] / batch.site_radius)**2)
            previousEpochXArrays = previousEpochXArrays.combine_first(returnXArray)
        return previousEpochXArrays

    def flatten_mean_results(data, mean_data=True):
        '''Returns the loss or time data either flattened or meaned along axis 1.
        Input: data, usually either loss or time data.
        output: flattened or meaned data
        '''
        # Note: This would be better implemented in Python 3.10's match/case method
        if (mean_data):
            return np.array(data).mean(axis=1)
        elif (not mean_data):
            return np.array(data).flatten()

    def get_layout_stats(dataset):
        '''Returns detailed information on the edges in a layout/dataset.
        Input: a single layout/dataset
        output: NDArray with the pwl, c_dist, n_dist

        For the purpose of evaluating the layout generation, transformation, and filter methods.
        '''
        LayoutEdgeStatsArray = np.zeros((3, len(dataset.y)))
        for i in range(len(dataset.y)):
            LayoutEdgeStatsArray[0][i] = abs(dataset.y[i])
            LayoutEdgeStatsArray[1][i] = math.sqrt((dataset.edge_attr[i][0])**2 + (dataset.edge_attr[i][1])**2)
            LayoutEdgeStatsArray[2][i] = math.sqrt((dataset.edge_attr[i][0] / dataset.site_radius)**2 + (dataset.edge_attr[i][1] / dataset.site_radius)**2)
        return LayoutEdgeStatsArray

    def print_layout_stats(LayoutEdgeStatsArray):
        '''Prints detailed information on the edges in a layout/dataset.
        Input: NDArray with pwl, c_dist, and n_dists => output from get_layout_stats
        output: prints the statistics

        For the purpose of evaluating the layout generation, transformation, and filter methods.
        '''
        for i, stats in enumerate(['pwl', 'c_dist', 'n_dist']):
            print(f'{stats}; Mean: {LayoutEdgeStatsArray[i].mean()}, stddev: {LayoutEdgeStatsArray[i].std()}, min: {LayoutEdgeStatsArray[i].min()}, max: {LayoutEdgeStatsArray[i].max()}')

    def plot_y_y(title, y_data_1, y_data_2, y_1_label, y_2_label, x_axis_label='Epoch', y_axis_label='loss', x_axis_min=None, x_axis_max=None, y_axis_min=None, y_axis_max=None):
        '''Plots a figure with 2 y_data sets
        '''
        plt.figure(layout='constrained')
        plt.plot(y_data_1, label=y_1_label)
        plt.plot(y_data_2, label=y_2_label)
        plt.xlabel(x_axis_label)
        plt.axis([x_axis_min, x_axis_max, y_axis_min, y_axis_max])
        plt.ylabel(y_axis_label)
        plt.title(title)
        plt.legend();

    def plot_x_y(title, x_data, y_data, y_label, x_axis_label='Epoch', y_axis_label='loss?', x_axis_min=None, x_axis_max=None, y_axis_min=None, y_axis_max=None):
        '''Plots a figure with an x, y dataset
        '''
        plt.figure(layout='constrained')
        plt.plot(x_data, y_data, label=y_label)
        plt.xlabel(x_axis_label)
        plt.axis([x_axis_min, x_axis_max, y_axis_min, y_axis_max])
        plt.ylabel(y_axis_label)
        plt.title(title)
        plt.legend();

    def plot_hist(title, y_data, y_label, bins, x_axis_label, y_axis_label, x_axis_min=None, x_axis_max=None, y_axis_min=None, y_axis_max=None, percent_format=True):
        '''Plots a histogram with a y dataset, either in percentage or occurences
        '''
        plt.figure(layout='constrained')
        if (percent_format):
            plt.hist(y_data, label=y_label, bins=bins, weights=np.ones(len(y_data)) / len(y_data))
        else:
            plt.hist(y_data, label=y_label, bins=bins)
        plt.xlabel(x_axis_label)
        plt.axis([x_axis_min, x_axis_max, y_axis_min, y_axis_max])
        plt.ylabel(y_axis_label)
        plt.title(title)
        plt.legend();
        if(percent_format):
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    def plot_2DHist(title, x_data, x_label, y_data, y_label, bins, x_axis_label, y_axis_label, x_axis_min=None, x_axis_max=None, y_axis_min=None, y_axis_max=None):
        '''Plots a 2D-histogram with an x, y dataset
        '''
        plt.figure(layout='constrained')
        raise NotImplementedError()