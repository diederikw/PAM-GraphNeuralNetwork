from torch_geometric.data import Dataset, Data
import torch
import os
import xarray
import numpy as np
import math

class PyGeoDataLoader(Dataset):
    '''
    Class which loads wflopg data into a PyTorch_Geometric dataset, including some transformation and filtering options.
    '''
    def __init__(self, root,
                wind_rose_reduction_method,
                filter_method,
                filter_threshold,
                seed=None,
                transform=None,
                pre_transform=None,
                pre_filter=None
                ):
        if (filter_method not in ['no_filter', 'random_filter', 'cartesian_filter', 'normalized_filter', 'pwl_filter']
            or wind_rose_reduction_method not in ['no_reduction', 'cardinal_reduction', 'prominent_reduction']):
            raise ValueError(f'The given filter_method: {filter_method} or wind_rose_reduction_method: {wind_rose_reduction_method} was not recognized. Please make sure it is one of the implemented options.')
        
        self.raw_file_names_set = os.listdir(os.path.join(root, 'raw'))
        self.processed_file_names_set = []
        for f in os.listdir(os.path.join(root, 'raw')):
            self.processed_file_names_set.append(f.replace('.nc', '.pt'))
        
        self.wind_rose_reduction_method = wind_rose_reduction_method
        self.filter_method = filter_method
        self.filter_threshold = filter_threshold

        if (seed is not None):
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng()

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_dir(self):
        return os.path.join(self.root, f'{self.wind_rose_reduction_method}-{self.filter_method}-{self.filter_threshold}')
    
    @property
    def raw_file_names(self):
        return self.raw_file_names_set

    @property
    def processed_file_names(self):
        return self.processed_file_names_set

    def download(self):
        pass

    def _process_one_step(self):
        '''
        This function is not used. 
        It could be good to process farm by farm using this function, but it produced many errors and a lot of work. 
        '''
        raise NotImplementedError("This function is not implemented")

    def filterEdges(self, x_target, y_target, x_source, y_source, site_radius, pwl):
        if (self.filter_method == 'pwl_filter' and pwl < self.filter_threshold):
            return False
        elif (self.filter_method == 'cartesian_filter' and math.sqrt(((x_target-x_source)*site_radius)**2 + ((y_target-y_source)*site_radius)**2) > self.filter_threshold):
            return False
        elif (self.filter_method == 'normalized_filter' and math.sqrt((x_target-x_source)**2 + (y_target-y_source)**2) > self.filter_threshold):
            return False
        elif (self.filter_method == 'random_filter' and self.filter_threshold < self.rng.uniform()):
            return False
        else:
            return True
    
    def nearestNeighbour(self, wind_dirs_original, wind_dirs_pmf_original, wind_dirs_new, halfway_values):
        wind_dirs_pmf_new = np.zeros((4))

        if (halfway_values[0] < wind_dirs_new[0]):
            v, w = 0, 3
        else:
            v, w = 1, 0
        # Note: This would be better implemented with Python 3.10's match/case method
        for i in range(wind_dirs_original.size):
            if (halfway_values[0] < wind_dirs_original[i] <= halfway_values[1]):
                wind_dirs_pmf_new[v] += wind_dirs_pmf_original[i]
            elif (halfway_values[1] < wind_dirs_original[i] <= halfway_values[2]):
                wind_dirs_pmf_new[v+1] += wind_dirs_pmf_original[i]
            elif (halfway_values[2] < wind_dirs_original[i] <= halfway_values[3]):
                wind_dirs_pmf_new[v+2] += wind_dirs_pmf_original[i]
            elif (halfway_values[3] < wind_dirs_original[i] <= 179 or -180 <= wind_dirs_original[i] <= halfway_values[0]):
                wind_dirs_pmf_new[w] += wind_dirs_pmf_original[i]
            else:
                raise ValueError(f'The wind direction given was: {wind_dirs_original[i]}, while it should fall between 0 and 359 (inclusive).')
        return wind_dirs_pmf_new

    def windReduction(self, wind_dirs_original, wind_dirs_pmf_original):
        prominent_ind = np.argpartition(wind_dirs_pmf_original, -1)[-1:]
        
        wind_dirs_new, halfway_values = np.zeros(4), np.zeros(4)
        if (self.wind_rose_reduction_method == 'cardinal_reduction'):
            wind_dirs_new[0], halfway_values[0] = np.array((0)), np.array((45))
        elif (self.wind_rose_reduction_method == 'prominent_reduction'):
            wind_dirs_new[0], halfway_values[0] = wind_dirs_original[prominent_ind], (wind_dirs_original[prominent_ind] + 45)
        elif (self.wind_rose_reduction_method == 'no_reduction'):
            raise ValueError("The windReduction method should not be called if the wind_rose_reduction_method is no_reduction.")
        else:
            raise ValueError()

        for i in range(4):
            if (i != 0):
                wind_dirs_new[i] = wind_dirs_new[0] + (90*i)
                halfway_values[i] = halfway_values[0] + (90*i)
            if (halfway_values[i] > 179):
                halfway_values[i] -= 360
            if (wind_dirs_new[i] > 179):
                    wind_dirs_new[i] -= 360

        wind_dirs_new.sort()
        halfway_values.sort()    
        wind_dirs_pmf_new = self.nearestNeighbour(wind_dirs_original, wind_dirs_pmf_original, wind_dirs_new, halfway_values)
        return wind_dirs_new, wind_dirs_pmf_new

    def process(self):
        idx = 0
        for file in self.raw_paths:
            # Load owflop of a single file / layout
            owflop = xarray.open_dataset(file)
            print(f'Starting with owflop: {owflop.name}')
            # Process this owflop into PyGeo shape
            # Note: This would be nicer with numpy arrays
            edge_index = []
            edge_attr = []
            y = []
            y_turbine = []

            power_turbine_waked = []
            power_turbine_wakeless = owflop.expected_wakeless_power.item()
            power_farm_waked = 0
            site_radius = owflop.site_radius.item()

            # Note: This should be implemented with Python 3.10's match/case method.
            if (owflop.wind_dirs_d.size != 4 and self.wind_rose_reduction_method != 'no_reduction'):
                wind_dirs_d, wind_dirs_pmf = self.windReduction(owflop.wind_dirs_d, owflop.wind_dirs_pmf)
                wind_dirs_r = np.deg2rad(wind_dirs_d)
            elif (owflop.wind_dirs_d.size == 4 or self.wind_rose_reduction_method == 'no_reduction'):
                wind_dirs_d, wind_dirs_pmf = np.array(owflop.wind_dirs_d), np.array(owflop.wind_dirs_pmf)
                wind_dirs_r = np.deg2rad(wind_dirs_d)
            else:
                raise ValueError()
            
            try:
                pdf_kappas = np.array(owflop.kappas)
                pdf_locs_r = np.array(owflop.locs_r)
                pdf_locs_d = np.rad2deg(pdf_locs_r).round(0)
                pdf_weights = np.array(owflop.weights)
            except:
                pdf_kappas = np.array((np.nan))
                pdf_locs_r = np.array((np.nan))
                pdf_locs_d = np.array((np.nan))
                pdf_weights = np.array((np.nan))

            for target in range(len(owflop.pwl.target)):
                target_pwl_total = 0
                for source in range(len(owflop.pwl.source)):
                    if (target != source and self.filterEdges(
                        owflop.layout.sel(target=target)[0].item(), owflop.layout.sel(target=target)[1].item(),
                        owflop.layout.sel(target=source)[0].item(), owflop.layout.sel(target=source)[1].item(),
                        site_radius, owflop.pwl.sel(target=target, source=source).item()
                    )):
                        edge_index.append([source, target])
                        edge_attr_list = [
                            (owflop.layout.sel(target=target)[0].item() - owflop.layout.sel(target=source)[0].item()) * site_radius,
                            (owflop.layout.sel(target=target)[1].item() - owflop.layout.sel(target=source)[1].item()) * site_radius
                        ]
                        edge_attr.append(edge_attr_list)
                        pwl = owflop.pwl.sel(target=target, source=source).item()
                        y.append([pwl])
                        target_pwl_total += pwl
                y_turbine.append([target_pwl_total])

                power_turbine_waked.append([power_turbine_wakeless - (power_turbine_wakeless * target_pwl_total)])
                power_farm_waked += power_turbine_wakeless - (power_turbine_wakeless * target_pwl_total)
            data = Data(
                x = torch.tensor(owflop.layout.values * site_radius, dtype=torch.float),
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr= torch.tensor(edge_attr, dtype=torch.float),

                y = torch.tensor(y, dtype=torch.float),
                y_turbine = torch.tensor(y_turbine, dtype=torch.float),         

                power_turbine_waked = torch.tensor(power_turbine_waked, dtype=torch.float),
                power_turbine_wakeless = torch.tensor(power_turbine_wakeless, dtype=torch.float),
                power_farm_waked = torch.tensor(round(power_farm_waked, 4), dtype=torch.float),
                site_radius = torch.tensor(site_radius, dtype=torch.float),
                
                wind_dirs_d = torch.tensor(wind_dirs_d, dtype=torch.float),
                wind_dirs_r = torch.tensor(wind_dirs_r, dtype=torch.float),
                wind_dirs_pmf = torch.tensor(wind_dirs_pmf, dtype=torch.float),
                # PDF parameters
                kappas = torch.tensor(pdf_kappas, dtype=torch.float),
                locs_r = torch.tensor(pdf_locs_r, dtype=torch.float),
                locs_d = torch.tensor(pdf_locs_d, dtype=torch.float),
                weights = torch.tensor(pdf_weights, dtype=torch.float)
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, self.processed_file_names[idx]))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data