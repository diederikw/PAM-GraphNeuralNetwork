# %%
import os
import torch
from torch_geometric.transforms import Compose, ToDevice

# Import data gen class
from create_data import OwflopDataGen as DataGenerator
# Import custom dataset class
from load_pygeo_data import PyGeoDataLoader
from load_pygeo_data import PyGeoDataLoader
#from load_pygeo_data_PDF import PyGeoDataLoader
'''
Settings begin
'''
# Generate data false or true
generate_owflop_data = False # TODO: Setting
load_to_pygeo_data = True # TODO: Setting
n_sites = 200 # TODO: Setting
single_WD = True # TODO: Setting
pdf_method = False # TODO: Setting

# General
# Set paths and options for experiment specific data
wind_rose_reduction_methods = ['no_reduction', 'cardinal_reduction', 'prominent_reduction']
filter_methods = ['no_filter', 'random_filter', 'cartesian_filter', 'normalized_filter', 'pwl_filter', 'random_filter']
filter_thresholds = [0, 0.8, 1.3, 0.5, 1e-3, 0.2]

wind_rose_reduction_methods = wind_rose_reduction_methods[2:3]
filter_methods = filter_methods[5:]
filter_thresholds = filter_thresholds[5:]

# Choose correct folder here
doc_path = r'C:\Studies\WFLO-GNN\documents'
# Load wind path
if (single_WD and pdf_method):
    data_path = os.path.join(doc_path, 'data', 'data_single_pdf_wd') # TODO: Setting
elif (single_WD and not pdf_method):
    data_path = os.path.join(doc_path, 'data', 'data_single_pmf_wd') # TODO: Setting
elif (not single_WD and pdf_method):
    data_path = os.path.join(doc_path, 'data', 'data_multi_wd') # TODO: Setting
else:
    raise ValueError()

'''
Settings end
'''
# Load other paths
raw_dir_path = os.path.join(data_path, 'raw')
pam_dir_path = os.path.join(data_path, 'pam')

# %%
# Set device for torch
if torch.cuda.is_available():
    print("Device is cuda")
    device = torch.device('cuda:0')
else:
    print("Device is cpu")
    device = torch.device('cpu')

# %%
# Data generation
# Set data generator's paths
if (generate_owflop_data):
    DataGenerator = DataGenerator(
        farm_dir=raw_dir_path, 
        pam_dir=pam_dir_path)

# %%
# Data generation
if (generate_owflop_data):
    for site_n in range(n_sites):
        DataGenerator.main(
            pdf_method=pdf_method,
            single_WD=single_WD
        )

# %%
if (single_WD and load_to_pygeo_data):
    wind_rose_reduction_method = 'no_reduction'
    
    for j, filter_method in enumerate(filter_methods):
        print(f'Starting dataloading with wind_reduction: {wind_rose_reduction_method}, and filter_method: {filter_method}')
        # PyGeo Dataset
        # Set data
        dataset = PyGeoDataLoader(
            root=data_path,
            wind_rose_reduction_method=wind_rose_reduction_method,
            filter_method=filter_method,
            filter_threshold=filter_thresholds[j],
            seed=None,
            transform=Compose([
                ToDevice(device)
            ])
        )
elif (load_to_pygeo_data):
    for i, wind_rose_reduction_method in enumerate(wind_rose_reduction_methods):
        for j, filter_method in enumerate(filter_methods):
            print(f'Starting dataloading with wind_reduction: {wind_rose_reduction_method}, and filter_method: {filter_method}')
            # PyGeo Dataset
            # Set data
            dataset = PyGeoDataLoader(
                root=data_path,
                wind_rose_reduction_method=wind_rose_reduction_method,
                filter_method=filter_method,
                filter_threshold=filter_thresholds[j],
                seed=None,
                transform=Compose([
                    ToDevice(device)
                ])
            )

else: 
    print('Data was not adapted/loaded into PyGeo standard')