# %% Imports
import numpy as np
import os
import math
import xarray
import wflopg
import copy
from numpy.random import default_rng
from wflopg import helpers
from wflopg import optimizers
from ruamel.yaml import YAML as _yaml
from scipy.stats import vonmises, rv_continuous
from typing import Optional

from PDFMixtureModel import PDFMixtureModel

def _yaml_load(f):
        return _yaml(typ='safe').load(f)

class OwflopDataGen():
    '''
    Class which creates wflopg data using pam calculated pwls'.
    '''
    def __init__(self, farm_dir, pam_dir, doc_dir=r'C:\Studies\WFLO-GNN\documents', rng=None, seed=None):
        self.farm_dir = farm_dir
        self.pam_dir = pam_dir
        self.doc_dir = doc_dir
        if (rng is None and seed is None):
            self.rng = default_rng()
        elif (rng is None and seed is not None):
            self.rng = default_rng(seed=seed)
        else: # rng != None
            self.rng = rng

    # %% Create PA Model
    def get_pa_model (self, num_dists, num_dirs, turbine_distance, site_radius, wind_resource):
        if 'pdf_parameters' in wind_resource:
            kappas = wind_resource['pdf_parameters']['kappas']
            locs_r = wind_resource['pdf_parameters']['locs_r']
            weights = wind_resource['pdf_parameters']['weights']
            pam_name = f'pam-SR={site_radius}-PDF-k={kappas}-l={locs_r}-w={weights}'
        else:
            pam_name = f'pam-SR={site_radius}-PMF'
        
        path = f'{self.pam_dir}\{pam_name}.nc'

        try: # Try to load PAM with correct parameters
            print(path)
            pa_model = xarray.open_dataarray(path)
            # Theta variable is changed for wflopg compatibility
            pa_model = pa_model.rename({'smallTheta': 'θ'})
        except: # If it doesn't exist, create PAM 

            # Load necessary files and create owflop
            os.chdir(self.doc_dir)
            o_pam = wflopg.Owflop()
            o_pam.load_problem("problem-pam-flexible.yaml",
                            layout={'type': "pam", 'num_dists': num_dists, 'num_dirs': num_dirs},
                            wake_model={'combination': "NO"},
                            site={'radius': site_radius},
                            turbine_distance=turbine_distance,
                            wind_resource=wind_resource)
            # transform to dask array to keep memory requirements ‘low’
            o_pam._ds = o_pam._ds.chunk({'target': 100})
            # compute PAM
            pa_model = wflopg.create_pam.compute(o_pam)
            pa_model = helpers.cyclic_extension(pa_model, 'θ', 2*np.pi)
            # transform the dask array back to normal xarray
            pa_model.load()
            # Save PAM to NetCDF file. Theta variable is changed for xarray compatibility
            pa_model = pa_model.rename({'θ': 'smallTheta'})
            try:
                xarray.DataArray.to_netcdf(pa_model, path = path)
            except:
                os.makedirs(self.pam_dir)
                xarray.DataArray.to_netcdf(pa_model, path = path)
            # Theta variable is changed for wflopg compatibility
            pa_model = pa_model.rename({'smallTheta': 'θ'})
        # Return Correct PAM
        return pa_model

    # %% Calculate the number of turbines 
    def calc_n_turbines (self, rotor_radius, site_radius, factor, ratio_value=30):
        if (factor < 0.5 or factor > 2.0):
            raise ValueError("The factor cannot be smaller than 0.5 or larger than 2.0")    
        else: 
            site_area = pow(site_radius, 2)*math.pi
            rotor_area = pow(rotor_radius*0.001, 2)*math.pi
            n_turbines_rough = (site_area / ratio_value) / rotor_area
            n_turbines = math.floor(factor*n_turbines_rough)
            return n_turbines

    # %% Create an optimized layout for an owflop/layout
    def optimize_farm (self, owflop, max_iterations=10, methods='abc', multiplier=1, scaling=[.8, 1.1], wake_spreading=False):
        # Optimize
        optimizers.step_iterator(owflop, max_iterations, methods, multiplier, scaling, wake_spreading)

    # %% Calculation of pwl with interpolation using pa_model
    def calc_pwl (self, layout, pam):
        xy = layout._ds.vector
        r = layout._ds.distance
        θ = np.arctan2(xy.sel(xy='y'), xy.sel(xy='x'))
        θ = θ.where(θ >= 0, θ + 2*np.pi) # ensure in [0, 2π[
        layout._ds['pwl'] = pam.interp(r=r, θ=θ, kwargs={'fill_value': np.inf})

    # %% Save layout data
    def save_layout(self, o_layout_ds, layout_name, file_path):  
        o_layout_ds.attrs["name"] = layout_name
        o_layout_ds = o_layout_ds.rename({'θ': 'smallTheta'})
        try:
            xarray.Dataset.to_netcdf(o_layout_ds, path = file_path)
        except:
            os.makedirs(self.farm_dir)
            xarray.Dataset.to_netcdf(o_layout_ds, path = file_path)
        # Theta variable is changed for wflopg compatibility
        o_layout_ds = o_layout_ds.rename({'smallTheta': 'θ'})

    # %% Create layouts function
    def get_layout_data (self, pam, site_radius, n_turbines_factor, site_violation_distance, wind_resource, optimise):
        # extract and process information and data from linked documents
        os.chdir(self.doc_dir)
        problem_doc = 'problem-farm-flexible.yaml' # This still exists from older version where the problem_doc differed for versions
        with open(problem_doc) as f:
            problem = _yaml_load(f)
        with open(problem['turbine']) as f: # TODO: Turbine could alter
            turbine = _yaml_load(f)
        rotor_radius = turbine['rotor_radius']
        # Calculate the amount of turbines based on parameters
        n_turbines = self.calc_n_turbines(rotor_radius=rotor_radius, site_radius=site_radius, factor=n_turbines_factor)

        # Set data destinations and names 
        directions = wind_resource['wind_rose']['directions']
        directions_pmf = wind_resource['wind_rose']['direction_pmf']
        if 'pdf_parameters' in wind_resource:
            kappas = wind_resource['pdf_parameters']['kappas']
            locs_r = wind_resource['pdf_parameters']['locs_r']
            weights = wind_resource['pdf_parameters']['weights']
            farm_name = f'farm-SR={site_radius}-NTF={n_turbines_factor}-SVD={site_violation_distance}-PDF-k={kappas}-l={locs_r}-w={weights}'
        else:
            farm_name = f'farm-SR={site_radius}-NTF={n_turbines_factor}-SVD={site_violation_distance}-PMF'
        if (optimise): farm_name = farm_name+"-o"
        file_path = f'{self.farm_dir}\{farm_name}.nc'

        try: # Try to load layout data with correct parameters
            o_layout_ds = xarray.open_dataset(file_path)
            # Theta variable is changed for wflopg compatibility
            o_layout_ds = o_layout_ds.rename({'smallTheta': 'θ'})
        except: # If it doesn't exist, create layout data 
            print(f'Creating layout: {farm_name}')
            
            # Load necessary files and create owflop
            os.chdir(self.doc_dir)
            o_layout = wflopg.Owflop()
            o_layout.load_problem(problem_doc,
                                layout={'type': "hex",
                                        'turbines': n_turbines,
                                        'site_violation_distance': site_violation_distance,
                                        'kwargs': {'randomize': True}
                                        },
                                site={'radius': site_radius},
                                wind_resource=wind_resource)
            
            o_layout.calculate_geometry()
            wflopg.create_layout.fix_constraints(o_layout)
            
            # Calculate pwl with pam, and save farms
            o_layout.calculate_wakeless_power()
            if (optimise): 
                self.optimize_farm(owflop=o_layout)
                o_layout.calculate_wakeless_power #Wakeless power needs to be calculated before and after optimisation
            self.calc_pwl(layout=o_layout, pam=pam)
            # Extra parameters for model
            o_layout._ds['site_radius'] = site_radius
            o_layout._ds['wind_dirs_d'] = directions
            o_layout._ds['wind_dirs_pmf'] = directions_pmf

            if 'pdf_parameters' in wind_resource:
                o_layout._ds['kappas'] = kappas
                o_layout._ds['locs_r'] = locs_r
                o_layout._ds['weights'] = weights
            o_layout_ds = o_layout._ds
            self.save_layout(o_layout_ds=o_layout_ds, layout_name=farm_name, file_path=file_path)
    
    # %% Create a wind_resource wind_rose
    def create_wind_resource_rose(self, pdf_method: bool, single_WD: bool, num_directions=120, pdf_model: Optional[rv_continuous] = None,):
        '''Create a wind_resource wind_rose
        '''
        if (pdf_method): # If we use a von mises pdf to calculate the wind resource pmf's
            if (360 % num_directions != 0):
                raise ValueError()
            directions_step_size = int(360 / num_directions)
            directions = np.arange(-180, 179, directions_step_size)
            direction_pmf = np.zeros(shape=(num_directions))
            speed_cpmf = np.ones(shape=(num_directions, 1))

            weights = []
            kappas = []
            locs_r = []
            if (single_WD and pdf_model is None): # If we need a newly generated single WD PDF
                mean, scale = 3, 8
                kappa_i = self.rng.wald(mean, scale)
                while (kappa_i > 9 or kappa_i == 0):
                    kappa_i = self.rng.wald(mean, scale)
                kappas.append(round(kappa_i, 2))
                rand_loc = self.rng.integers(-180, 180)
                locs_r.append(round(np.deg2rad(rand_loc), 3))
                weights.append(1.0)
                mixed_model = vonmises(kappa=kappas, loc=locs_r)
            elif (not single_WD and pdf_model is None):  # If we need a newly generated multi WD mixed PDF
                n_prim_wind_dir = 3#rng.integers(2, 4)
                single_dir_pdf_models = []
                forbidden_locs = set()
                range_n = 45
                for i in range(n_prim_wind_dir):
                    rand_loc = self.rng.integers(-180, 180)
                    while rand_loc in forbidden_locs:
                        rand_loc = self.rng.integers(-180, 180)
                    if (rand_loc-range_n < -180):
                        forbidden_locs = forbidden_locs.union(set(range(180-abs(rand_loc-range_n), 179)))
                        forbidden_locs = forbidden_locs.union(set(range(-180, rand_loc+range_n)))
                    elif (rand_loc+range_n >= 180):
                        forbidden_locs = forbidden_locs.union(set(range(rand_loc-range_n, 179)))
                        forbidden_locs = forbidden_locs.union(set(range(-180, range_n-(179 - rand_loc))))
                    else:
                        forbidden_locs = forbidden_locs.union(set(range(rand_loc-range_n, rand_loc+range_n)))
                    locs_r.append(round(np.deg2rad(rand_loc), 3))
                    mean, scale = 3, 8
                    kappa_i = self.rng.wald(mean, scale)
                    while (kappa_i > 9 or kappa_i == 0):
                        kappa_i = self.rng.wald(mean, scale)
                    kappas.append(round(kappa_i, 2))
                    single_dir_pdf_models.append(vonmises(kappa=kappas[i], loc=locs_r[i]))
                    weight_i = self.rng.uniform(1, 5)
                    weights.append(round(weight_i, 2))
                mixed_model = PDFMixtureModel(single_dir_pdf_models, weights)
            elif (pdf_model is not None): # If a PDF model is given
                mixed_model = pdf_model
            else:
                raise ValueError(f'The combination of pdf_method:{pdf_method}, single_WD:{single_WD}, and pdf_model:{pdf_model}, is not a valid option.')
            
            
            direction_pmf[0] = mixed_model.cdf(np.deg2rad((directions[0] + directions[1]) / 2)) - mixed_model.cdf(np.deg2rad(directions[0]))
            direction_pmf[0] += mixed_model.cdf(np.deg2rad(179)) - mixed_model.cdf(np.deg2rad((180 + directions[-1]) / 2))
            direction_pmf[-1] = mixed_model.cdf(np.deg2rad((180 + directions[-1]) / 2)) - mixed_model.cdf(np.deg2rad((directions[-2] + directions[-1]) / 2))
            for i in range(1, len(directions)-1):
                direction_pmf[i] = mixed_model.cdf(np.deg2rad((directions[i] + directions[i+1]) / 2)) - mixed_model.cdf(np.deg2rad((directions[i-1] + directions[i]) / 2))
            direction_pmf = direction_pmf / direction_pmf.sum()
        
            wind_resource = {
                'wind_rose': {
                    "directions": directions, "direction_pmf": direction_pmf, "speeds": [9.8], "speed_cpmf": speed_cpmf
                },
                'pdf_parameters': {
                    "kappas": kappas, "locs_r": locs_r, "weights": weights
                }
            }
        else: # Else we are considering a 100% single direction
            if (not single_WD):
                raise ValueError()
            wind_resource = {
                'wind_rose': {
                    "directions": [0],"direction_pmf": [1],"speeds": [9.8],"speed_cpmf": [[1]]
                }
            }
        return wind_resource

    # %% Main method
    def main (         
            self, 
            pdf_method: bool, single_WD: bool,
            wind_resource: dict=None,
            site_radius_low=1.4, site_radius_high=4.5,
            n_turbines_factor_start_value=0.75, n_turbines_factor_end_value = 2.0, 
            site_violation_distance_start_value=0, site_violation_distance_end_value = 0.02, 
            optimise_chance = True,
            pa_dists = 100, pa_dirs = 360, pa_td = 2
    ):
        '''
        Set the parameters using randomisation.
        A PAM is created, using those parameters.
        Create a windfarm including PAM-calculated pwl's.
        '''
        # Create random parameters for farm generation
        # Random site radius, rounded to 2 decimals
        site_radius = round(self.rng.uniform(low=site_radius_low, high=site_radius_high), 2)
            
        if (optimise_chance):
            optimise = self.rng.choice(a=[True, False])
        else: 
            optimise = False

        # Random factor
        n_turbines_factor = round(self.rng.uniform(low=n_turbines_factor_start_value, high=n_turbines_factor_end_value), 2)
        # Random site violation distance
        site_violation_distance = round(self.rng.uniform(low=site_violation_distance_start_value, high=site_violation_distance_end_value), 3)

        if (not wind_resource):
            wind_resource = self.create_wind_resource_rose(pdf_method, single_WD)
        # Have to create a copy as the original wind_resource gets deleted in wflopg init (possibly there, not 100% debugged)
        wind_resource_copy = copy.deepcopy(wind_resource)
        # Get or create PAM
        pam = self.get_pa_model(num_dists=pa_dists, num_dirs=pa_dirs, turbine_distance=pa_td, site_radius=site_radius, wind_resource=wind_resource)
        # Create layaout, save in storage/farm folder
        self.get_layout_data(pam=pam, site_radius=site_radius, n_turbines_factor=n_turbines_factor, site_violation_distance=site_violation_distance, wind_resource=wind_resource_copy, optimise=optimise) # Create or retrieve the layout