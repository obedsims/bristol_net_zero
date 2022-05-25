import calliope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bristol_utils import cap_loc_score_potential, cap_loc_calc, energy_loc_calc, energycon_loc_calc, \
    storage_cap_loc_calc, capacity_factor
import seaborn as sns

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)

# We increase logging verbosity
calliope.set_log_verbosity('INFO', include_solver_output=False)

'''
------------------------------------------------------------------------------
---------------Scenarios---------------(Min Cost & Maximise Renewable make-up)
------------------------------------------------------------------------------
'''

'''
---------------Scenario---------------(100% Renewable fraction, CCGT w/CCS)
'''

# Define model
# decarb_model = calliope.Model('/bristol_model/model.yaml', scenario='r100,ccgt_ccs_on')

# Run the model
# decarb_model.run()

decarb_model = calliope.read_netcdf('bristol_model/results/r100_ccs_noimports.nc')

# print(decarb_model.results)

plot = decarb_model.plot.timeseries(array='results', subset={'costs': ['monetary']})
data = decarb_model.get_formatted_array('carrier_con').loc[{'carriers': 'electricity'}].sum(['locs']).to_pandas()

locations = decarb_model.inputs.locs.values

# Plot a specific site
decarb_model.plot.timeseries(array='results', subset={'costs': ['monetary'], 'locs': ['Feeder Rd 11kv']})

# Plot the transmission/distribution network
decarb_model.plot.transmission(
    mapbox_access_token='xxx')

# Total system cost
total_decarb_cost = abs(decarb_model.get_formatted_array('cost').loc[{'costs': 'monetary'}].sum(['locs', 'techs']).to_pandas())

tech_array = decarb_model.inputs.techlists.values
print(tech_array[0])
tech_list = tech_array[0].split(",")

# Calculation of the capacity installed in each location
decarb_cap_per_loc = cap_loc_calc(decarb_model, techs=tech_list)
# Add total installed capacity to the dataframe
columns = ['wind_onshore_new', 'pv_rooftop_new', 'EfW_new', 'utility_solar_new', 'tidal_stream']
decarb_cap_per_loc['total_new_installed_cap'] = decarb_cap_per_loc[columns].sum(axis=1)
decarb_cap_per_loc.to_clipboard(sep=',')

energy_cap_max_list = decarb_model.inputs.energy_cap_max.values
loc_techs_list = decarb_model.inputs.loc_techs.values
energy_cap_max = pd.DataFrame(data=[energy_cap_max_list], columns=loc_techs_list)

# Calculation of the storage capacity installed in each location (kWh)
# storage_cap_per_loc = storage_cap_loc_calc(model, techs=tech_list)
decarb_storage_cap_per_loc = decarb_model.get_formatted_array('storage_cap').loc[
    {'techs': ['utility_battery', 'small_scale_battery']}].to_pandas()
# Add total storage capacity per region column to dataframe
decarb_storage_cap_per_loc['total_storage_cap'] = decarb_storage_cap_per_loc.sum(axis=1)

storage_cap_max_list = decarb_model.inputs.storage_cap_max.values
loc_storage_list = decarb_model.inputs.loc_techs_store.values
storage_cap_max = pd.DataFrame(data=[storage_cap_max_list], columns=loc_storage_list)

lcoes = decarb_model.results.systemwide_levelised_cost.loc[{'carriers': 'electricity', 'costs': 'monetary'}].to_pandas()
lcoes.plot(kind='bar')

'''
---------------SPORES---------------
'''

scenarios = {'scenario_20': 'r100,ccgt_ccs_on,green_ppa_on,spores'}

i = 1
for key, value in scenarios.items():
    print(f"{key}: {value}")
    model = calliope.Model('/bristol_model/model.yaml',
                           scenario=f'{value}')
    # Run the model
    model.run()
    model.to_netcdf('bristol_model/results/scenario_%d.nc' % i)
    i += 1

# Define model
scenario_1 = calliope.read_netcdf('/bristol_model/results/scenario_1.nc')

cap_ratio_spores = []
stor_ratio_spores = []

for i in range(0, len(scenario_1.results.spores.to_pandas())):
    a = scenario_1.get_formatted_array('energy_cap').loc[{'spores': i}].sum(['locs']).to_pandas()
    b = scenario_1.get_formatted_array('energy_cap_max').sum(['locs']).to_pandas()
    c = scenario_1.get_formatted_array('storage_cap').loc[{'spores': i, 'techs': ['utility_battery', 'small_scale_battery']}].sum(['locs']).to_pandas()
    d = scenario_1.get_formatted_array('storage_cap_max').loc[{'techs': ['utility_battery', 'small_scale_battery']}].sum(['locs']).to_pandas()
    cap_ratio = a / b
    stor_ratio = c / d
    cap_ratio = cap_ratio.rename('Spore %d' % i)
    stor_ratio = stor_ratio.rename('Spore %d' % i)
    cap_ratio_spores.append(cap_ratio)
    stor_ratio_spores.append(stor_ratio)


cap_df = pd.concat(cap_ratio_spores, axis=1)
stor_df = pd.concat(stor_ratio_spores, axis=1)

cap_df2 = cap_df[cap_df.index.str.contains("hvac") == False]
cap_df2 = cap_df2.drop(index=['demand_electricity', 'EfW_new', 'ccgt_existing', 'pv_rooftop', 'utility_solar', 'wind_onshore'],
             axis=0)
cap_df2.update(stor_df)          # Update the battery rows with the values from stor_df
cap_df2 = cap_df2.transpose()
cap_optimal = cap_df2.head(1)

sns.set_style("white")
sns.set_context("paper")
f, ax = plt.subplots(1, 1, figsize=(16, 10))
labels = ['EfW', 'CCGT w/CCS', 'Green PPA Import', 'Rooftop PV', 'Small-scale BESS', 'Tidal Stream', 'Utility-scale BESS',
          'Utility-scale Solar', 'Onshore Wind']
sns.stripplot(data=cap_df2, ax=ax, jitter=0.05, color='indianred', alpha=0.5)
sns.stripplot(data=cap_optimal, ax=ax, jitter=0.05, color='black')
sns.violinplot(data=cap_df2, ax=ax, color='grey', linewidth=0.2, scale='width')
plt.setp(ax.collections, alpha=.4)
ax.set_xticklabels(labels)
plt.ylim(0, 1)
ax.set_ylabel('Utilisation of Potential Capacity Expansion', fontsize=20)
ax.set_xlabel('')


'''
---------------COST-OPTIMAL SPORE---------------
'''

# Calculate the capacity factor of the AC transmission
opt_capfac = scenario_1.get_formatted_array('capacity_factor').loc[{'carriers': 'electricity', 'spores': 0}].mean(
    'timesteps').to_pandas()
opt_capfac = opt_capfac.loc[:, opt_capfac.columns.str.contains("hvac")]
opt_capfac.to_clipboard(sep=',')

tech_list = scenario_1.inputs.techlists.values
tech_list = tech_list[0].split(",")
tech_list = tech_list[0:10]
storage_list = ['small_scale_battery', 'utility_battery']
tech_list = tech_list

empty_df = pd.DataFrame()

for i in tech_list:
    cap_over_time = scenario_1.get_formatted_array('carrier_prod').loc[{'carriers': 'electricity', 'spores': 0, 'techs': i}].to_pandas()
    empty_df = empty_df.add(cap_over_time, fill_value=0)

# Calculate the total AC distribution amount
opt_trans1_total = scenario_1.get_formatted_array('carrier_prod').loc[{'spores': 0}].sum(['locs', 'timesteps']).to_pandas()
opt_trans2_total = scenario_1.get_formatted_array('carrier_con').loc[{'spores': 0}].sum(['locs', 'timesteps']).to_pandas()

f2, ax = plt.subplots(1, 1, figsize=(16, 10))
sns.set_style("white")
sns.set_context("paper")
sns.barplot(data=opt_trans1_total/1e6, color="dodgerblue", ax=ax, orient='h', label='carrier_prod')
sns.barplot(data=opt_trans2_total/1e6, color="indianred", ax=ax, orient='h', label='carrier_con')
sns.despine(left=True, bottom=True)

# Calculate the firm capacity of each primary substation
firm_cap_per_loc = scenario_1.get_formatted_array('energy_cap_max').sum(['locs']).to_pandas()
firm_cap_per_loc = firm_cap_per_loc.loc[firm_cap_per_loc.index.str.contains("hvac")]
firm_cap_per_loc.to_clipboard(sep=',')

print(scenario_1.inputs.techs)

scenario_1.plot.timeseries(array='results', subset={'costs': ['monetary'], 'spores': [32]})
scenario_1.plot.timeseries(array='results', subset={'costs': ['monetary'], 'spores': [0], 'locs': ['Eastville']})

# Calculate the amount of renewable imports required
total_imports_spores = []

for i in range(0, len(scenario_1.results.spores.to_pandas())):
    import_total = scenario_1.get_formatted_array('carrier_prod').loc[{'spores': i, 'techs': 'green_ppa_import'}].sum(['locs', 'timesteps']).to_pandas()
    import_total = import_total.rename('Spore %d' % i)
    total_imports_spores.append(import_total)

total_import_df = pd.concat(total_imports_spores, axis=1)
total_import_df = total_import_df.transpose()/1e6
