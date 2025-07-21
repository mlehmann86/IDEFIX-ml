from matplotlib.gridspec import GridSpec
import vtk_io
import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from scipy.ndimage import uniform_filter1d

import os
sys.path.append(os.getenv("IDEFIX_DIR"))

import pandas as pd

'''
utilities
'''

def GetLinearGrowth(time, data, ginterval):
    g1, g2 = ginterval
    fit = np.polyfit(time[g1:g2+1], np.log(data[g1:g2+1]), 1)
    return fit[0]


'''
plotting parameters
'''
fontsize = 24
nlev = 128
nclev = 6
cmap = plt.cm.inferno


def ReadAnalysisFile(filename):
    # Read the file into a pandas DataFrame
    df = pd.read_csv(filename, delim_whitespace=True)
    
    # Convert each column to a 1D array (list)
    t       = df['t'].values/(2.0*np.pi) #convert to orbits
    dVmax   = df['dVmax'].values
    vx2     = df['vx2'].values
    vy2     = df['vy2'].values
    vz2     = df['vz2'].values
    vx      = df['vx'].values
    vy      = df['vy'].values
    vz      = df['vz'].values
    
    return t, dVmax, vx2, vy2, vz2, vx, vy, vz

def PlotMaxEvol1D(loc, tinterval=(0, 10), var='vg', yrange=None, logscale=True, avg=1):
   
    time_orbits, max_dvg, vx2, vy2, vz2, vx, vy, vz = ReadAnalysisFile(loc+'/analysis.dat')

    if var == 'vg':
        data = np.copy(max_dvg)
        ylab = r'$max|\delta v|$'
   
    if avg > 1:  # perform a running time average of avg grid points wide
        data = uniform_filter1d(data, size=avg)

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    plt.xlim(0.0, np.amax(time_orbits))
    if yrange != None:
        plt.ylim(yrange[0], yrange[1])

    if logscale == True:
        plt.yscale('log')

    plt.plot(time_orbits, data, linewidth=2)

    plt.rc('font', size=fontsize, weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='lower right', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(r'$t/P$', fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    plt.ylabel(ylab, fontsize=fontsize)

    fname = loc+'/idefix_maxevol_'+var
    plt.savefig(fname, dpi=150)


def PlotCompareWithDedalus(loc, var='vg', yrange=None,
                         logscale=True, avg=1):

    #Read Idefix analysis 
    time_orbits, max_dvg, vx2, vy2, vz2, vx, vy, vz = ReadAnalysisFile(loc+'/analysis.dat')

    #Read corresponding Dedalus analysis, the file must be in the same loc and named Dedalus_analysis.txt
    analysis_dedalus    = np.loadtxt(loc+'/analysis_dedalus.txt', delimiter=',')
    time_orbits_dedalus = analysis_dedalus[:, 0]/(2.0*np.pi)
    max_dvg_dedalus     = analysis_dedalus[:, 1]
    if avg > 1:  # perform a running time average of avg grid points wide
        max_dvg_dedalus = uniform_filter1d(max_dvg_dedalus, size=avg)


    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    if not yrange:
        plt.ylim(1e-4, 1e-1)
    else:
        plt.ylim(yrange[0], yrange[1])

    if logscale == True:
        plt.yscale('log')

    plt.plot(time_orbits, max_dvg, linewidth=2, label='Idefix')
    plt.plot(time_orbits_dedalus, max_dvg_dedalus, linewidth=2, label='Dedalus')

    if var == 'vg':
        ylabel = r'$max|\delta v|$'
  
    plt.rc('font', size=fontsize, weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='lower right', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(r'$t/P$', fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    plt.ylabel(ylabel, fontsize=fontsize)

    # ax.set_title(title, weight='bold')

    fname = 'idefix_vs_dedalus'
    plt.savefig(fname, dpi=150)

def PlotMaxEvol1DCompare(locs, valfven2, labels, title, fname, var='vg', xrange=None, yrange=None,
                         logscale=True, avg=1, growth_theory=None, gtheory_labels=None, dgnorm=False):

    # reference alfven speed for each case, stored in an array for later use
    va = [np.sqrt(va2) for va2 in valfven2]

    '''
    Compare 1D evolution across cases
    '''

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    if not yrange:
        plt.ylim(1e-4, 1e-1)
    else:
        plt.ylim(yrange[0], yrange[1])

    if logscale == True:
        plt.yscale('log')

    tend = 0
    for i, loc in enumerate(locs):
        analysis = np.loadtxt(loc+'/analysis.txt', delimiter=',')
        time = analysis[:, 0]
        max_dvg = analysis[:, 1]/va[i]
        max_eps = analysis[:, 2]
        if dgnorm == True:
            max_eps /= max_eps[0]  # normalize by initial value
        rms_vg = analysis[:, 3]
        rms_vgx = analysis[:, 4]
        rms_vgy = analysis[:, 5]
        rms_vgz = analysis[:, 6]
        amflux = analysis[:, 7]
        amfluxd = analysis[:, 8]
        min_eps = analysis[:, 9]

        time_orbits = np.copy(time)/(2.0*np.pi)  # convert to orbits
        if var == 'vg':
            data = max_dvg
        elif var == 'epsilon':
            data = max_eps
        elif var == 'rms':
            data = rms_vg
        elif var == 'rmsx':
            data = rms_vgx
        elif var == 'rmsy':
            data = rms_vgy
        elif var == 'rmsz':
            data = rms_vgz
        elif var == 'amflux':
            data = amflux
        elif var == 'amfluxd':
            data = amfluxd
        elif var == 'min_eps':
            data = min_eps

        if avg > 1:  # perform a running time average of avg grid points wide
            data = uniform_filter1d(data, size=avg)

        # if pert == True:
        #     data-=data[0]
        if i == 1:
            next(ax._get_lines.prop_cycler)  # skip the next color in the cycle
        plt.plot(time_orbits, data, linewidth=2, label=labels[i])
        # else:
        #     plt.plot(time_orbits, data, linewidth=2, label=labels[i])#, linestyle="dashed")

        if growth_theory != None and var == 'vg':  # and i == len(locs)-1:
            theory_curve = data[0]*np.exp((time-time[0])*growth_theory[i])
            if gtheory_labels != None:
                lab = gtheory_labels[i]
            else:
                lab = 'theory'
            # plt.plot(time_orbits, theory_curve, linestyle="dashed",markersize=8, label=lab, markevery=int(len(time_orbits)/10))
            plt.plot(time_orbits, theory_curve, linestyle="", marker="X",
                     markersize=8, label=lab, markevery=int(len(time_orbits)/10))
        maxorbits = np.amax(time_orbits)
        tend = np.amax([maxorbits, tend])

    if not xrange:
        plt.xlim(0.0, tend)
    else:
        plt.xlim(xrange[0], xrange[1])

    if var == 'vg':
        ylabel = r'$max|\delta v|/V_A$'
    elif var == 'epsilon':
        if dgnorm == False:
            ylabel = r'$max(\epsilon)$'
        else:
            ylabel = r'$max(\epsilon)/\epsilon_0$'
    elif var == 'rms':
        ylabel = r'$rms(\delta v)$'
    elif var == 'rmsx':
        ylabel = r'$rms(\delta v_{x})$'
    elif var == 'rmsy':
        ylabel = r'$rms(\delta v_{y})$'
    elif var == 'rmsz':
        ylabel = r'$rms(\delta v_{z})$'
    elif var == 'amflux':
        ylabel = r'$\overline{\delta v_{dx} \delta v_{gy}}$'
    elif var == 'amfluxd':
        ylabel = r'$\overline{\delta v_{dx} \delta v_{dy}}$'
    elif var == 'min_eps':
        ylabel = r'$min(\epsilon)$'

    if var == 'min_eps':
        plt.axhline(y=0, color='r', linestyle='dotted')

    plt.rc('font', size=fontsize, weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='upper left', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(r'$t/P$', fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    plt.ylabel(ylabel, fontsize=fontsize)

    ax.set_title(title, weight='bold')

    fname = 'hallSI_plot_compare_'+var+'_'+fname
    plt.savefig(fname, dpi=150)


def Plot2DContour(loc, output_n, var,
                  cut='xz', arange=None, plotrange=None,
                  log=None, pert=False, title=None, aspect=None):

    # Read initial state and grid info
    output0 = vtk_io.readVTK(loc+'data.0000.vtk')
    x = output0.x
    y = output0.y
    z = output0.z

    # Read file
    output_nstr = "{:04d}".format(output_n)
    fname = loc+'data.'+output_nstr+'.vtk'
    output = vtk_io.readVTK(fname)

    # Get timestamp in orbits
    tstamp = output.t[0]/(2.0*np.pi)

    # Get data at the timestamp and read axis information
    if var == 'rho':
        vname = 'RHO'
        clabel = r'$\rho$'
    elif var == 'pressure':
        vname = 'PRS'
        clabel = r'P'
    elif var == 'vx':
        vname = 'VX1'
        clabel = r'$v_x$'
    elif var == 'vz':
        vname = 'VX3'
        clabel = r'$v_z$'

    data3d = output.data[vname]

    # Compute fractional perturbation if desired
    if pert == True:
        data3d0 = output0.data[vname]
        data3d /= data3d0
        data3d -= 1.0
        ylabel = r'$\Delta$' + ylabel

    # Get slice info
    if cut == 'xz':
        xaxis = x
        yaxis = z
        xlabel = r'$x/H_g$'
        ylabel = r'$z/H_g$'
        data2d = np.mean(data3d, axis=1)

    if cut == 'xy':
        xaxis = x
        yaxis = y
        xlabel = r'$x/H_g$'
        ylabel = r'$y/H_g$'
        data2d = np.mean(data3d, axis=2)

    if cut == 'yz':
        xaxis = y
        yaxis = z
        xlabel = r'$y/H_g$'
        ylabel = r'$z/H_g$'
        data2d = np.mean(data3d, axis=0)

    if log != None:
        data2d = np.log10(data2d)

    xmin, xmax = -np.amax(xaxis), np.amax(xaxis)
    ymin, ymax = -np.amax(yaxis), np.amax(yaxis)

    if aspect == None:
        aspect = (ymax-ymin)/(xmax-xmin)

    if plotrange == None:
        minv = np.amin(data2d)
        maxv = np.amax(data2d)
    else:
        minv = plotrange[0]
        maxv = plotrange[1]

    levels = np.linspace(minv, maxv, nlev)
    clevels = np.linspace(minv, maxv, nclev)

    plt.rc('font', size=fontsize, weight='bold')

    #figsize = (7,5)
    figsize = (10, 3)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cp = plt.contourf(xaxis, yaxis, np.transpose(data2d), levels, cmap=cmap)

    cbar = plt.colorbar(cp, ax=ax, ticks=clevels,
                        format='%.2f', pad=0.005, shrink=0.695)
    # cbar    = plt.colorbar(cp,ticks=clevels,format='%.1f',pad=0)

    cbar.set_label(clabel)
    ax.set_box_aspect(aspect)
    # ax.set_aspect('equal', adjustable='box')

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.xaxis.set_major_locator(plt.MaxNLocator(1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)  # ,labelpad=-1)

    if title != None:
        ax.set_title(title, weight='bold')
    else:
        ax.set_title(r't={0:3.0f}'.format(tstamp)+"P", weight='bold')

    # plt.savefig(loc+'idefixSI_'+var+'2D_'+output_nstr,dpi=150)
    plt.savefig(loc+var+'_'+cut+'2D_'+output_nstr, dpi=150)
    plt.close()


def Plot1DProfile(loc, output_n, var, profile='x',
                  plotrange=None, pert=False, title=None):

    # Read initial state and grid info
    output0 = vtk_io.readVTK(loc+'data.0000.vtk')
    xaxis = output0.x
    yaxis = output0.y
    zaxis = output0.z

    # Read desired output
    output_nstr = "{:04d}".format(output_n)
    fname = loc+'data.'+output_nstr+'.vtk'
    output = vtk_io.readVTK(fname)

    # Get timestamp in orbits
    tstamp = output.t[0]/(2.0*np.pi)

    # Get data at the timestamp and read axis information
    if var == 'rho':
        vname = 'RHO'
        ylabel = r'$\rho$'
    elif var == 'pressure':
        vname = 'PRS'
        ylabel = r'$P$'
    elif var == 'vx':
        vname = 'VX1'
        ylabel = r'$v_x$'

    data3d = output.data[vname]

    # Compute fractional perturbation if desired
    if pert == True:
        data3d0 = output0.data[vname]
        data3d /= data3d0
        data3d -= 1.0
        ylabel = r'$\Delta$' + ylabel

    if profile == 'x':  # average over z and y to get x profile
        data1d = np.mean(data3d, axis=(1, 2))
        xlabel = r'$x/H_\text{g}$'
        xmin, xmax = -np.amax(xaxis), np.amax(xaxis)
    elif profile == 'y':
        data1d = np.mean(data3d, axis=(0, 2))
        xlabel = r'$y/H_\text{g}$'
        xmin, xmax = -np.amax(yaxis), np.amax(yaxis)
    elif profile == 'z':
        data1d = np.mean(data3d, axis=(0, 1))
        xlabel = r'$z/H_\text{g}$'
        xmin, xmax = -np.amax(zaxis), np.amax(zaxis)

    # if log != None:
    # 	data1d = np.log10(data2d)

    plt.rc('font', size=fontsize, weight='bold')

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    plt.rc('font', size=fontsize, weight='bold')

    plt.plot(xaxis, data1d, linewidth=2)
    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='upper left', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(xlabel, fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    plt.ylabel(ylabel, fontsize=fontsize)

    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)#,labelpad=-1)

    if title != None:
        ax.set_title(title, weight='bold')
    else:
        ax.set_title(r't={0:3.0f}'.format(tstamp)+"P", weight='bold')

    plt.savefig(loc+var+'_'+profile+'1D_'+output_nstr, dpi=150)
    plt.close()


def Plot1DBuoyancy(loc, output_n,
                   plotrange=None, title=None):

    # assume gamma=1.4!
    gamma=1.4

    # Read desired output
    output_nstr = "{:04d}".format(output_n)
    fname = loc+'data.'+output_nstr+'.vtk'
    output = vtk_io.readVTK(fname)

    xaxis = output.x
    tstamp = output.t[0]/(2.0*np.pi)

    rho = output.data['RHO']
    pressure = output.data['PRS']

    rho1d = np.mean(rho, axis=(1, 2))
    pressure1d = np.mean(pressure, axis=(1, 2))

    dPdx = np.gradient(pressure1d, xaxis)
    dlnPdx = np.gradient(np.log(pressure1d), xaxis)
    dlnrhodx = np.gradient(np.log(rho1d), xaxis)
    Nsq = -(1.0/rho1d)*dPdx*(dlnPdx/gamma - dlnrhodx)

    ylabel = r'$N^2$'
    xlabel = r'$x/H_\text{g}$'
    xmin, xmax = -np.amax(xaxis), np.amax(xaxis)

    plt.rc('font', size=fontsize, weight='bold')

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    plt.rc('font', size=fontsize, weight='bold')

    plt.plot(xaxis, Nsq, linewidth=2)
    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='upper left', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(xlabel, fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    plt.ylabel(ylabel, fontsize=fontsize)

    if title != None:
        ax.set_title(title, weight='bold')
    else:
        ax.set_title(r't={0:3.0f}'.format(tstamp)+"P", weight='bold')

    plt.savefig(loc+'Nsq_1D_'+output_nstr, dpi=150)
    plt.close()


if __name__ == "__main__":
    loc = './'
    k = 9
    var = 'vx'

    PlotMaxEvol1D(loc, var='vg', logscale=True)
    PlotCompareWithDedalus(loc, var='vg', yrange=[1e-4,1], logscale=True)

    Plot1DProfile(loc, k, 'rho', pert=False)
    Plot1DProfile(loc, k, 'pressure', pert=False)
    Plot1DProfile(loc, k, 'vx', pert=False)
    Plot1DBuoyancy(loc, k)

    # for n in range(1, k+1):
    Plot2DContour(loc, k, 'vx', cut='xz', pert=False)




    
