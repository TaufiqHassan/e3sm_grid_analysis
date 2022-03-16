#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 20:10:14 2022

@author: thassan
"""
from scipy.interpolate import griddata
from cartopy.util import add_cyclic_point
from subprocess import run
import os
from matplotlib import pyplot, patches, collections
from cartopy import crs, feature
from xarray import open_dataset, broadcast
import numpy as np
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.tri as tri
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import LinearSegmentedColormap
import cmaps as cm
from shapely import geometry
from matplotlib import patches, collections

def get_data(ds, variable_name):
    if variable_name in ds.variables.keys():
        data = ds[variable_name]
    elif variable_name == 'PRECT':
        precc = get_data(ds, 'PRECC')
        precl = get_data(ds, 'PRECL')
        data = precc + precl
        data.attrs = precc.attrs
        data.attrs['long_name'] = 'Total precipitation rate'
    else:
        raise NameError('%s not found in dataset'%variable_name)

    # Adjust units
    if variable_name in ('PRECC', 'PRECL', 'PRECT'):
        if data.attrs['units'].lower() == 'm/s':
            attrs = data.attrs
            data = 60 * 60 * 24 * 1e3 * data
            data.attrs = attrs
            data.attrs['units'] = 'mm/day'
    return data

def fix_longitudes(lon):
    return lon.assign_coords(lon=((lon + 180) % 360) - 180) #np.where(lon > 180, lon - 360, lon))


def plot_map(lon, lat, data, ax = None, vm=.5e10, lon_corners=None, eq=None,lat_corners=None, method='contourf',**kwargs):

    
    diff=(vm-data.min())/16
    l=np.arange(data.min(),vm,diff)
    if sum(n < 0 for n in l)==0:
        sdiff=(l[1]-l[0])/4
        sl=np.arange(l[0],l[1],sdiff)
        l2=np.append(sl,l[1:])
        mdiff=(l[-1]-l[-2])/4
        ml=np.arange(l[-2],l[-1],mdiff)
        lev=np.append(l2[:-2],ml).tolist()
        # c='Spectral_r'
        interval = np.hstack([np.linspace(0.15, 1)])
        colors = cm.WhiteBlueGreenYellowRed(interval)
        c = LinearSegmentedColormap.from_list('name', colors)
        e='max'
    else:
        sdiff=(data.max()-0)/9
        sl=np.arange(0,data.max(),sdiff)
        ml=-1*sl
        lev=np.append(ml[::-1],sl[1:]).tolist()
        # c='YlOrRd'
        interval = np.hstack([np.linspace(0.15, 1)])
        colors = cm.WhiteBlueGreenYellowRed(interval)
        c = LinearSegmentedColormap.from_list('name', colors)
        e='both'
    # lev=[0.0, 20371677184.0, 40743354368.0, 61115031552.0, 81486708736.0, 162973417472.0, 244460126208.0, 325946834944.0, 407433543680.0, 488920252416.0, 570406961152.0, 651893669888.0, 733380378624.0, 814867087360.0, 896353796096.0, 977840504832.0, 1059327213568.0]
    # Setup plot axes
    ax.set_global()
    ax.coastlines(lw=1.5, resolution='50m')
    ax.set_extent((-100,-50,20,65), crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = gl.ylabels_right = False
    gl.xlines = gl.ylines = False
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}

    # Make plot
    if all([v is not None for v in (lon_corners, lat_corners)]):
        pl = plot_map_native(lon_corners, lat_corners, data)
    elif 'ncol' in data.dims:
        new_lon = lon.copy(deep=True)
        new_lon.values = np.where(lon > 180, lon - 360, lon)
        # Drop missing data
        new_lon = new_lon.where(data.squeeze().notnull()).dropna('ncol')
        lat = lat.where(data.squeeze().notnull()).dropna('ncol')
        data = data.squeeze().dropna('ncol')

        # Plot
        if method == 'pcolor':
            pl = ax.tripcolor(new_lon.squeeze(), lat.squeeze(), data.squeeze(),cmap=c,alpha=1,edgecolors='k',lw=0.001,transform=ccrs.PlateCarree(),vmax=vm,**kwargs)
        elif method == 'contourf':
            print(vm)
            pl = ax.tricontourf(new_lon.squeeze(), lat.squeeze(), data.squeeze(),transform=ccrs.PlateCarree(),edgecolors='k',cmap=c,extend='max',levels=lev,**kwargs)
        else:
            raise ValueError('%s not a valid plot method'%(method))
    else:
        new_lon = lon.assign_coords(lon=((lon + 180) % 360) - 180)
        # print(data.shape,lon.shape,lat.shape)
        if eq!=None:
            new_lon=new_lon.round(1)
        data2, lon2 = add_cyclic_point(data.values, coord=new_lon,axis=1)
        if method == 'pcolor':
            pl = ax.pcolormesh(lon2, lat, data2,edgecolors='k',cmap=c,alpha=1,lw=0.005,transform=ccrs.PlateCarree(),vmax=vm)
        elif method == 'contourf':
            pl = ax.contourf(lon2, lat, data2, edgecolors='k',lw=.05,levels=lev,cmap=c,extend=e,robust=True)
        else:
            raise ValueError('%s not a valid plot method'%(method))
    cbar=plt.colorbar(pl, orientation='horizontal', pad=0.05) 
    cbar.set_label(label='BC surface emission (molecules/$cm^{2}$/s)',size=15)
    # Return plot handle
    return pl

def plot_field(dataset, field, ax=None, method='contourf', plot_type='map'):

    # Get data
    data = get_data(dataset, field).mean(dim='time',keep_attrs=True)
    # data = get_data(dataset, field)[1]

    if plot_type == 'map':
        lon = get_data(dataset, 'lon')
        lat = get_data(dataset, 'lat')
        pl = plot_map(lon, lat, data, ax=ax, method=method)

    return pl

def zonal_mean(data, weights=None):
    # Compute mean
    if weights is not None:
        weights, data = xr.broadcast(weights, data)
        zonal_mean = (weights * data).sum(dim='lon', keep_attrs=True) / weights.sum(dim='lon', keep_attrs=True)
    else:
        zonal_mean = data.mean(dim='lon')

    # Copy attributes
    zonal_mean.attrs = data.attrs

    return zonal_mean

def plot_time_series(data, **kwargs):
    ax = plt.gca()
    if len(data.shape) == 1:
        pl = ax.plot(data['time'].values, data, **kwargs)
        ax.set_ylabel(f'{data.long_name} ({data.units})')
        ax.set_xlabel('Time')
    else:
        pl = ax.contourf(data['time'].values, data.lev, data.transpose(), **kwargs)
        ax.set_ylabel(f'Vertical level')
        ax.set_xlabel('Time')
        cb = plt.colorbar(
            pl, orientation='horizontal',
            label=ax.set_ylabel(f'{data.long_name} ({data.units})')
        )
    plt.xticks(rotation=45)
    return pl


def get_area_weights(ds):
    # Get weights; either use pre-computed or cosine(latitude) weights
    if 'area' in ds.variables.keys():
        wgt = ds.area
    else:
        # Use xarray.ufuncs to work on lazily evaluated dask arrays
        y = get_data(ds, 'lat')
        wgt = xr.ufuncs.cos(y * np.pi / 180.0)
    return wgt

def area_average(data, weights, dims=None, **kwargs):
    
    '''Calculate area-weighted average over dims.'''
      
    if dims is None: dims = data.dims
        
    # Need to broadcast weights to make sure they have the
    # same size/shape as data. For example, data is (lat, lon)
    # but we passed weights with shape (lat,), or data is
    # (time, ncol) but we passed weights with shape (ncol,).
    weights, *__ = broadcast(weights, data)
    
    # Mask weights consistent with data so we do not miscount
    # missing columns
    weights = weights.where(data.notnull())
    
    # Do the averaging        
    data_mean = (weights * data).sum(dim=dims, **kwargs) / weights.sum(dim=dims, **kwargs)
    
    # Copy over attributes, which we lose in the averaging
    # calculation
    data_mean.attrs = data.attrs
    
    # Return averaged data
    return data_mean

def replace_outliers(data, dim=0, perc=0.99):

  # calculate percentile 
  threshold = data[dim].quantile(perc)

  # find outliers and replace them with max among remaining values 
  mask = data[dim].where(abs(data[dim]) <= threshold)
  max_value = mask.max().values
  # .where replace outliers with nan
  mask = mask.fillna(max_value)
  print(mask)
  data[dim] = mask

  return data

def add_region_box(ax, region=[-45, 0, 51, 60],alpha=0.3, col='blue', ec=None):
    """ """
    geom = geometry.box(minx=region[0],maxx=region[1],miny=region[2],maxy=region[3])
    ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor=col, edgecolor=ec, alpha=alpha,lw=3)
    return ax


# Func to plot the exodus files
def get_lines(ds,ax):
    x = ds['coord'][0,:].squeeze()
    y = ds['coord'][1,:].squeeze()
    z = ds['coord'][2,:].squeeze()
    lon = np.arctan2(y, x) * 180.0 / np.pi
    lat = np.arcsin(z) * 180.0 / np.pi
    corner_indices = ds['connect1']
    xx = lon[corner_indices[:,:] - 1]
    yy = lat[corner_indices[:,:] - 1]
    lines = [[[xx[i,j], yy[i,j]] for j in range(xx.shape[1])] for i in range(xx.shape[0])]
    line_collection = collections.LineCollection(lines, transform=crs.Geodetic(),colors='k',linewidth=0.1)
    ax.add_collection(line_collection)
    
 #C9BD9E   