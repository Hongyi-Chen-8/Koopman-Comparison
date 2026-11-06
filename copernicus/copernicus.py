import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import copernicusmarine
import xarray as xr
import numpy as np
from scipy.ndimage import gaussian_filter
from datetime import datetime
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_ocean(data_dict, filename="ocean_kinematics.mp4", fps=8):
    """
    Generates a synchronized MP4 animation of all available ocean variables.
    Optimized for memory efficiency by updating plot objects in-place.
    """
    timestamps = list(data_dict.keys())
    if not timestamps:
        print("\033[31mError: Empty data dictionary provided to animator.\033[0m")
        return

    first_frame = data_dict[timestamps[0]]
    lon, lat = first_frame['lon'], first_frame['lat']
    
    # 1. Determine spatial layout based on available variables
    active_vars = ['flow']
    if 'temp' in first_frame: active_vars.append('temp')
    if 'chl' in first_frame: active_vars.append('chl')
    
    n_plots = len(active_vars)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), dpi=200)
    if n_plots == 1: 
        axes = [axes] # Ensure iterable

    print(f"--> Pre-computing global bounds for {len(timestamps)} frames...")
    
    # 2. Compute Global Extents for Static Colormaps
    bounds = {}
    
    # Flow limits
    all_mag = [np.sqrt(d['u']**2 + d['v']**2) for d in data_dict.values()]
    bounds['flow'] = (0, np.nanmax(all_mag))
    
    if 'temp' in active_vars:
        all_t = [d['temp'] for d in data_dict.values()]
        bounds['temp'] = (np.nanmin(all_t), np.nanmax(all_t))
        
    if 'chl' in active_vars:
        all_c = [d['chl'] for d in data_dict.values()]
        # Chl often has severe outliers; 98th percentile is standard for visual clarity
        bounds['chl'] = (0, np.nanpercentile(all_c, 98))

    # 3. Initialize Plot Objects (Frame 0)
    plot_objs = {}
    
    # Flow Setup
    ax_f = axes[active_vars.index('flow')]
    im_f = ax_f.pcolormesh(lon, lat, all_mag[0], cmap='YlGnBu_r', shading='nearest',
                           vmin=bounds['flow'][0], vmax=bounds['flow'][1])
    plt.colorbar(im_f, ax=ax_f, label='Velocity (m/s)')
    
    # Adaptive quiver striding based on mesh density
    stride_y, stride_x = max(1, lon.shape[0] // 25), max(1, lon.shape[1] // 25)
    q_f = ax_f.quiver(lon[::stride_y, ::stride_x], lat[::stride_y, ::stride_x], 
                      first_frame['u'][::stride_y, ::stride_x], first_frame['v'][::stride_y, ::stride_x], 
                      color='white', alpha=0.75, scale_units='xy')
    ax_f.set_title("Ocean Kinematics")
    plot_objs['flow'] = (im_f, q_f)

    # Temp Setup
    if 'temp' in active_vars:
        ax_t = axes[active_vars.index('temp')]
        im_t = ax_t.pcolormesh(lon, lat, first_frame['temp'], cmap='magma', shading='nearest',
                               vmin=bounds['temp'][0], vmax=bounds['temp'][1])
        plt.colorbar(im_t, ax=ax_t, label='Temperature (°C)')
        ax_t.set_title("Thermodynamic State")
        plot_objs['temp'] = im_t

    # Chl Setup
    if 'chl' in active_vars:
        ax_c = axes[active_vars.index('chl')]
        im_c = ax_c.pcolormesh(lon, lat, first_frame['chl'], cmap='viridis', shading='nearest',
                               vmin=bounds['chl'][0], vmax=bounds['chl'][1])
        plt.colorbar(im_c, ax=ax_c, label='Chl-a (mg/m³)')
        ax_c.set_title("Biogeochemical (Chl-a)")
        plot_objs['chl'] = im_c

    title = fig.suptitle(f"Epoch: {timestamps[0]}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 4. Animation Update Loop (O(1) Memory Replacement)
    def update(frame_idx):
        t_str = timestamps[frame_idx]
        d = data_dict[t_str]
        
        # In-place C-array replacement
        plot_objs['flow'][0].set_array(all_mag[frame_idx].ravel())
        plot_objs['flow'][1].set_UVC(d['u'][::stride_y, ::stride_x], d['v'][::stride_y, ::stride_x])
        
        if 'temp' in plot_objs:
            plot_objs['temp'].set_array(d['temp'].ravel())
            
        if 'chl' in plot_objs:
            plot_objs['chl'].set_array(d['chl'].ravel())
            
        title.set_text(f"Epoch: {t_str}")
        return [] # Blitting disabled for multi-axis pcolormesh compatibility

    print(f"--> Rendering {len(timestamps)} frames to {filename}...")
    
    anim = animation.FuncAnimation(fig, update, frames=len(timestamps), blit=False)
    
    # Requires FFmpeg installed on system
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000, codec="h264")
    anim.save(filename, writer=writer)
    
    plt.close(fig)
    print(f"\033[32m-- Compilation Complete: {filename} --\033[0m\n")

# --- DATASET CONFIGURATION ---
# Note: ANFC is used for chlorophyll when date > 2026-02-28
MY_BGC = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
ANFC_BGC = "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m"

DATASETS = {
    'flow': {"id": "cmems_mod_glo_phy-cur_anfc_0.083deg_PT1H-m", "vars": ['uo', 'vo']},
    'temp': {"id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT1H-m", "vars": ['thetao']},
    'chlorophyll': {"vars": ['chl']} # ID selected dynamically
}

def fetch_ocean_subset(mode, extent, date_str, depth_limit=100):
    conf = DATASETS[mode]
    z_min, z_max = 1.0, (depth_limit if mode == 'flow' else 1.0)
    
    dataset_id = conf.get("id")
    
    # Dynamic selection for Chlorophyll
    if mode == 'chlorophyll':
        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
        cutoff_dt = datetime(2026, 2, 28)
        dataset_id = MY_BGC if target_dt <= cutoff_dt else ANFC_BGC
        print(f"   Using BGC Dataset: {'Multi-Year' if dataset_id == MY_BGC else 'Analysis/Forecast'}")

    return copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        minimum_longitude=extent[0],
        maximum_longitude=extent[1],
        minimum_latitude=extent[2],
        maximum_latitude=extent[3],
        start_datetime=f"{date_str}T00:00:00",
        end_datetime=f"{date_str}T23:59:59",
        minimum_depth=z_min,
        maximum_depth=z_max,
        variables=conf['vars']
    )

def plot_ocean_state(mode, extent, date, reference_ds=None):
    print(f"--> Processing {mode}...")
    ds_to_return = None
    try:
        ds = fetch_ocean_subset(mode, extent, date)
        ds = ds.isel(time=0).compute()
        actual_time = np.datetime_as_string(ds.time.values, unit='D')
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Standard dimension mapping to handle BGC/PHY naming differences
        lat_n = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_n = 'longitude' if 'longitude' in ds.coords else 'lon'
        d_n = 'depth' if 'depth' in ds.dims else 'depth' # usually consistent in 2026 versions

        if mode == 'flow':
            ds_avg = ds.mean(dim='depth')
            mag = np.sqrt(ds_avg.uo**2 + ds_avg.vo**2)
            im = ax.pcolormesh(ds_avg[lon_n], ds_avg[lat_n], mag, cmap='YlGnBu_r', shading='auto')
            # ax.quiver(ds_avg[lon_n][::5], ds_avg[lat_n][::5], 
            #           ds_avg.uo[::5, ::5], ds_avg.vo[::5, ::5], color='white', alpha=0.6)
            plt.colorbar(im, ax=ax, label='Velocity (m/s)')
            ds_to_return = ds 

        elif mode == 'temp':
            temp_data = ds.thetao.isel({d_n: 0})
            im = ax.pcolormesh(ds[lon_n], ds[lat_n], temp_data, cmap='magma', shading='auto')
            plt.colorbar(im, ax=ax, label='Temperature (°C)')
            
        elif mode == 'chlorophyll':
            chl_raw = ds.chl.isel({d_n: 0}).rename({lat_n: 'latitude', lon_n: 'longitude'})

            if reference_ds is not None:
                # 1. High-Res Physics Mask (0.083 deg)
                high_res_mask = ~np.isnan(reference_ds.uo.isel(depth=0))
                
                # 2. Cubic Interp + Gaussian (0.5)
                chl_filled = chl_raw.interpolate_na(dim='longitude', method='nearest', fill_value='extrapolate')
                chl_interp = chl_filled.interp(latitude=reference_ds.latitude, 
                                               longitude=reference_ds.longitude, 
                                               method="linear").fillna(0)
                
                smoothed_values = gaussian_filter(chl_interp.values, sigma=0.5)
                chl_final_vals = np.where(high_res_mask, smoothed_values, np.nan)
                chl_final = xr.DataArray(chl_final_vals, coords=chl_interp.coords, dims=chl_interp.dims)
            else:
                chl_final = chl_raw

            im = ax.pcolormesh(chl_final.longitude, chl_final.latitude, chl_final, cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax, label='Chl-a (mg/m3)')

        ax.set_title(f"Global Ocean {mode.capitalize()} | Data Date: {actual_time}")
        plt.tight_layout()
        plt.savefig(f"images/{mode}_{actual_time.replace('-', '')}.png", dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"CRITICAL ERROR in {mode}: {e}")
        
    return ds_to_return

def pull_copernicus(extent, vars, time_query=None, depth=1, target_pts = 10000):
    translate = {
        'u': 'uo',
        'v': 'vo',
        'temp': 'thetao',
        'chl': 'chl'
    }
    # target_pts = 10000 
    dt = 6 # Hours

    vars = [translate[var] for var in vars]
    print(vars)

    output = {}
    print(f"=== Pulling Data ===")

    ## Clamp Extent strictly to fix dataset bounds exceed warning ----
    extent = list(extent)
    extent[2] = max(-80.0, float(extent[2]))
    extent[3] = min(80.0,  float(extent[3]))

    ## Setup Time Query --------------------------------------------------------------------------
    dates = []
    times = []
    for raw_t in time_query:
        if 'T' in raw_t:
            d, t = raw_t.split('T')
            dates.append(d)
            times.append(t + ":00" if len(t.split(':')) == 2 else t)
        else:
            split = raw_t.split('-')
            if len(split) == 3:
                dates.append(raw_t)
                times.append("00:00:00" if len(dates) == 1 else "23:59:59")
            elif len(split) == 4:
                dates.append(f"{split[0]}-{split[1]}-{split[2]}")
                times.append(f"{split[3]}:00:00")
            elif len(split) >= 5:
                dates.append(f"{split[0]}-{split[1]}-{split[2]}")
                times.append(f"{split[3]}:{split[4]}:00")

    ## Dataset IDs --------------------------------------------------------------------------------
    history = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
    forecast = "cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m"

    datasets = {
        'flow': {"id": "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m", "vars": ['uo', 'vo']},
        'temp': {"id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i", "vars": ['thetao']},
        'chlorophyll': {"vars": ['chl']} 
    }

    ## Loading Primary Dataset --------------------------------------------------------------------
    conf = datasets['flow']
    z_min, z_max = 0.49402499198913574, 0.49402499198913574 
    dataset_id = conf.get("id")

    var_test = [var for var in vars if var not in ['chl', 'thetao']]

    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        minimum_longitude=extent[0],
        maximum_longitude=extent[1],
        minimum_latitude=extent[2],
        maximum_latitude=extent[3],
        start_datetime=f"{dates[0]}T{times[0]}",
        end_datetime=f"{dates[-1]}T{times[-1]}",
        minimum_depth=z_min,
        maximum_depth=z_max,
        variables=var_test
    )
    
    ## Primary Subset & Load ----------------------------------------------------------------------
    total_pts = ds.longitude.size * ds.latitude.size
    if total_pts > target_pts:
        step = int(np.ceil(np.sqrt(total_pts / target_pts)))
        ds = ds.isel(latitude=slice(0, None, step), longitude=slice(0, None, step))
        print(f"Subsampled grid with step {step}. New shape: {ds.latitude.size}x{ds.longitude.size} ({ds.latitude.size*ds.longitude.size} pts)")

    # OPTIMIZATION A: Filter Time and Depth before network payload transfer
    time_strs = np.datetime_as_string(ds.time.values, unit='h')
    time_indices = [i for i, t in enumerate(time_strs) if int(t.split('T')[1].split(':')[0]) % dt == 0]
    
    ds = ds.isel(time=time_indices)
    if 'depth' in ds.dims:
        ds = ds.mean(dim='depth')

    print('Loading Primary Data (Network Fetch)...')
    ds = ds.load()
    print('Data Loaded.')

    ## Temp Setup ---------------------------------------------------------------------------------
    if 'thetao' in vars:
        print('Temperature Data Queried...')
        temp_ds = copernicusmarine.open_dataset(
            dataset_id=datasets['temp']["id"],
            minimum_longitude=extent[0],
            maximum_longitude=extent[1],
            minimum_latitude=extent[2],
            maximum_latitude=extent[3],
            start_datetime=f"{dates[0]}T00:00:00",
            end_datetime=f"{dates[-1]}T23:59:59",
            minimum_depth=z_min,
            maximum_depth=z_max,
            variables=['thetao']
        )
        
        # Temp has 6H resolution, while our flow grid is 1H. Interpolate time nearest.
        print('Interpolating Temperature time grid...')
        temp_ds = temp_ds.interp(time=ds.time, method='nearest')
        
        if total_pts > target_pts:
            temp_ds = temp_ds.isel(latitude=slice(0, None, step), longitude=slice(0, None, step))

        print('Loading Temperature Data...')
        temp_ds = temp_ds.load()
        print('Temperature Data Loaded.')

    ## Chlorophyll Setup --------------------------------------------------------------------------
    if 'chl' in vars:
        print('Chlorophyll Data Queried...')
        target_dt = datetime.strptime(dates[-1], "%Y-%m-%d")
        cutoff_dt = datetime(2026, 2, 28)
        chl_dataset_id = history if target_dt <= cutoff_dt else forecast
        print(f"   Using BGC Dataset: {'Multi-Year' if chl_dataset_id == history else 'Analysis/Forecast'}")

        chl_ds = copernicusmarine.open_dataset(
            dataset_id=chl_dataset_id,
            minimum_longitude=extent[0],
            maximum_longitude=extent[1],
            minimum_latitude=extent[2],
            maximum_latitude=extent[3],
            start_datetime=f"{dates[0]}T00:00:00",
            end_datetime=f"{dates[-1]}T23:59:59",
            minimum_depth=0.5057600140571594, 
            maximum_depth=0.5057600140571594,
            variables=['chl']
        )

        if 'depth' in chl_ds.dims:
            chl_ds = chl_ds.mean(dim='depth')

        chl_lat_n = 'latitude' if 'latitude' in chl_ds.coords else 'lat'
        chl_lon_n = 'longitude' if 'longitude' in chl_ds.coords else 'lon'
        chl_ds = chl_ds.rename({chl_lat_n: 'latitude', chl_lon_n: 'longitude'})

        # OPTIMIZATION C: Vectorized interpolation to the physics grid prior to loop
        print('Interpolating Chlorophyll spatial grid...')
        chl_ds = chl_ds.interpolate_na(dim='longitude', method='nearest', fill_value='extrapolate')
        chl_ds = chl_ds.interp(latitude=ds.latitude, longitude=ds.longitude, method='linear').fillna(0)
        
        print('Interpolating Chlorophyll time grid...')
        chl_ds = chl_ds.interp(time=ds.time, method='nearest')

        print('Loading Chlorophyll Data...')
        chl_ds = chl_ds.load()
        print('Chlorophyll Data Loaded.')

    ## OPTIMIZATION B: Hoist Loop Invariants ------------------------------------------------------
    lon_1d = np.array(ds.longitude.values)
    lat_1d = np.array(ds.latitude.values)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    m_to_deg = 1.0 / 111132.0
    sec_to_hr = 3600.0
    
    # Pre-compute conversion matrices
    v_conv = sec_to_hr * m_to_deg
    u_conv = sec_to_hr * m_to_deg / np.cos(np.radians(lat_2d))
    
    # Base physics mask used for Chl logic
    base_flow_mask = None 

    ## Loop Data Extraction -----------------------------------------------------------------------
    actual_time = np.datetime_as_string(ds.time.values, unit='h').tolist()
    
    for idx, timestamp_str in enumerate(actual_time):
        date, time_str = timestamp_str.split('T')
        hr = time_str.split(':')[0]
        timestamp = f"{date}-{hr}"
        
        print('\nExtracting:', timestamp)

        subset = ds.isel(time=idx)

        # Pull raw arrays directly using NumPy compiled code
        u_raw = np.nan_to_num(subset.uo.values).reshape(lon_2d.shape)
        v_raw = np.nan_to_num(subset.vo.values).reshape(lon_2d.shape)

        if base_flow_mask is None:
            base_flow_mask = ~np.isnan(subset.uo.values).reshape(lon_2d.shape)

        u = u_raw * u_conv
        v = v_raw * v_conv

        output[timestamp] = {
            'lon': lon_2d,
            'lat': lat_2d,
            'u': u,
            'v': v,
            'u_raw': u_raw,   # m/s — for visual/particle display
            'v_raw': v_raw    # m/s — for visual/particle display
        }

        if 'thetao' in vars:
            temp = np.nan_to_num(temp_ds.isel(time=idx).thetao.values).reshape(lon_2d.shape)
            output[timestamp]['temp'] = temp

        if 'chl' in vars:
            # Data is already interpolated to 0.083deg grid
            chl_raw_vals = chl_ds.isel(time=idx).chl.values.reshape(lon_2d.shape)
            
            # Apply Gaussian filter and spatial masking locally
            chl_filtered = gaussian_filter(chl_raw_vals, sigma=0.5)
            chl_final = np.where(base_flow_mask, chl_filtered, 0)
            
            output[timestamp]['chl'] = chl_final

    return output

if __name__ == "__main__":
    #study_area = [-74, -68, 35, 40] 
    study_area = [143, 148, -45, -40]
    # target_date = "2025-03-10-06" 
    
    # phy_ref = plot_ocean_state('flow', study_area, target_date)
    # plot_ocean_state('temp', study_area, target_date)
    # plot_ocean_state('chlorophyll', study_area, target_date, reference_ds=phy_ref)

    times = ["2025-1-1", "2025-7-1"]
    #vars = ['u', 'v', 'temp', 'chl']
    vars = ['u', 'v']

    import os

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('videos', exist_ok=True)


    output = pull_copernicus(study_area, vars, time_query=times)

    with open('outputs/flow.pkl', 'wb') as f:
        pickle.dump(output, f)
        print("Flow data saved to outputs/flow.pkl")

    animate_ocean(output, filename='videos/ocean.mp4')

    # if 'temp' in vars or 'chl' in vars:
    #     pull_myocean()
    # else:
    #     pull_HYCOM()