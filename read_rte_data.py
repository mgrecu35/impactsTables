import netCDF4 as nc

def read_data():
    with nc.Dataset('rte_CoSSIR_15Jan2023.nc') as f:
        kext=f.variables['kext'][:]
        scat=f.variables['scat'][:]
        asym=f.variables['asym'][:]
        temp=f.variables['temp'][:]
        tb=f.variables['tb'][:]

    return kext, scat, asym, temp, tb