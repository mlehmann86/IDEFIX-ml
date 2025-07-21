from matplotlib.gridspec import GridSpec
import sys
import numpy as np
#from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from scipy.ndimage import uniform_filter1d

import os
sys.path.append(os.getenv("IDEFIX_DIR"))
sys.path.append(os.path.abspath('/Users/minkailin/Projects/idefix-mkl/setups/COS-GPU'))

#import vtk_io from pytools in idefix
current_dir  = os.path.dirname(os.path.abspath(__file__))  # Directory of COS_plot.py
project_root = os.path.abspath(os.path.join(current_dir, '../../'))  # Adjust based on depth
pytools_dir  = os.path.join(project_root, 'pytools')
# Add the pytools directory to Python's module search path
sys.path.insert(0, pytools_dir)
import vtk_io

import pandas as pd
import importlib.util

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

def ReadAnalysisFile(loc, var='vg', dedalus=False, avg = 1):
    #default is to read idefix data
    if dedalus == False:
        # Read the file into a pandas DataFrame
        df = pd.read_csv(loc+'/analysis.dat', sep=r"\s+")

        # Convert each column to a 1D array (list)
        t       = df['t'].values/(2.0*np.pi) #convert to orbits
        dvmax   = df['dVmax'].values
        vx2     = df['<vx2>'].values
        vy2     = df['<dvy2>'].values
        vz2     = df['<vz2>'].values
        amf     = df['<amf>'].values

        rms_vx  = np.sqrt(vx2)
        rms_vy  = np.sqrt(vy2)
        rms_vz  = np.sqrt(vz2)

    else: #otherwise read dedalus file
        analysis_dedalus    = np.loadtxt(loc+'/analysis.txt', delimiter=',')
        t                   = analysis_dedalus[:, 0]/(2.0*np.pi)
        dvmax               = analysis_dedalus[:, 1]
        rms_vx              = analysis_dedalus[:, 2]
        rms_vy              = analysis_dedalus[:, 3]
        rms_vz              = analysis_dedalus[:, 4]
        amf                 = analysis_dedalus[:, 5]
        hflux               = analysis_dedalus[:, 6]

    if var == 'vg':
        data = dvmax
    elif var == 'rmsx':
        data = rms_vx
    elif var == 'rmsy':
        data = rms_vy
    elif var == 'rmsz':
        data = rms_vz
    elif var == 'amf':
        data =  amf
    elif var == '-amf':
        data = -amf #return negative AMF

    if avg > 1:  # perform a running time average of avg grid points wide
        data = uniform_filter1d(data, size=avg)

    return t, data

def PlotMaxEvol1D(loc, tinterval=(0, 10), var='vg', yrange=None, logscale=True, avg=1, gtheory=None):

    t, data1d  = ReadAnalysisFile(loc, var = var)

    tphysical = t*2.0*np.pi

    # if (var == 'vg') or (var == 'rmsx'):

    data = np.copy(data1d)

    if var == 'vg':
        ylab = r'$max|\delta v|$'
    elif var == 'rmsx':
        ylab = r'$rms(v_x)$'
    elif var == 'rmsy':
        ylab = r'$rms(\delta v_y)$'
    elif var == 'rmsz':
        ylab = r'$rms(v_z)$'

    t1, t2 = tinterval
    g1     = np.argmin(np.abs(t-t1))
    g2     = np.argmin(np.abs(t-t2))
    ginterval = (g1, g2)
    growth_sim  = GetLinearGrowth(tphysical, data, ginterval) #t is in orbits, need to convert to real time, P=2*pi

    print("{0:^32}".format("************"))
    print("{0:^32}".format("Growth rates"))
    print("{0:^32}".format("************"))
    print("simulation = {0:17.15e}".format(growth_sim))
    if gtheory != None:
            print("theory     = {0:17.15e}".format(gtheory))

    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True)
    ax = fig.add_subplot()

    plt.xlim(0.0, np.amax(t))
    if yrange != None:
        plt.ylim(yrange[0], yrange[1])

    if logscale == True:
        plt.yscale('log')

    plt.plot(t, data, linewidth=2)
    if gtheory != None:
        theory_curve = data[g1]*np.exp((tphysical-tphysical[g1])*gtheory)
        plt.plot(t[0:2*g2], theory_curve[0:2*g2], color='black', linestyle="dashed", label="theory")

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

def PlotMaxEvol1DCompare(locs, labels, dlocs=False, dlabels=False, var='vg', xrange=None, yrange=None,
                         logscale=True, avg=1):

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

    #plot idefix results
    for i, loc in enumerate(locs):
        t, data  = ReadAnalysisFile(loc, var=var, avg=avg)
        plt.plot(t, data, linewidth=2, label=labels[i])

    #plot dedalus results
    if dlocs != False:
        # plt.gca().set_prop_cycle(None)
        for j, loc in enumerate(dlocs):
            t, data = ReadAnalysisFile(loc, var=var, avg=avg, dedalus=True)
            plt.plot(t, data, linewidth=2, linestyle='dashed', label=dlabels[j])

    if not xrange:
        plt.xlim(np.amin(t), np.amax(t))
    else:
        plt.xlim(xrange[0], xrange[1])

    if var == 'vg':
        ylabel = r'$max|\delta v|$'
    elif var == 'rmsx':
        ylabel = r'$rms(v_{x})$'
    elif var == 'rmsy':
        ylabel = r'$rms(\delta v_{y})$'
    elif var == 'rmsz':
        ylabel = r'$rms(v_{z})$'
    elif var == 'amf':
        ylabel = r'$AMF$'
    elif var == '-amf':
        ylabel = r'$-AMF$'

    plt.rc('font', size=fontsize, weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend = ax.legend(lines1, labels1, loc='lower right', frameon=False,
                       ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize, weight='bold')
    plt.xlabel(r'$t/P$', fontsize=fontsize)

    plt.yticks(fontsize=fontsize, weight='bold')
    plt.ylabel(ylabel, fontsize=fontsize)

    fname = 'idefix_plot_compare_'+var
    plt.savefig(fname, dpi=150)

def Plot2DContour(loc, output_n, var,
                  cut='xz', arange=None, plotrange=None,
                  log=None, pert=False, title=None, aspect=None, slice=None, nonaxi=False):

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
    elif var == 'vy':
        vname = 'VX2'
        clabel = r'$v_y$'
    elif var == 'vz':
        vname = 'VX3'
        clabel = r'$v_z$'
    elif var == 'vort':
        clabel = r'$\omega_z$'

    if var != 'vort':
        data3d = output.data[vname]
    else:
        vx = output.data['VX1']
        vy = output.data['VX2']

        dvx_dy = np.gradient(vx, y, axis=1)  # Partial derivative of vx with respect to y
        dvy_dx = np.gradient(vy, x, axis=0)  # Partial derivative of vy with respect to x

        omega_z = dvy_dx - dvx_dy

        # omega_z_avg_y = np.mean(omega_z, axis=1, keepdims=True)

        data3d = omega_z #- omega_z_avg_y

    if nonaxi == True: #take out the axisymmetric component

        data3d_axi = np.mean(data3d, axis=1, keepdims=True)
        data3d    -= data3d_axi

        clabel = r'$\Delta$'+clabel

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
        if slice == None:
            data2d = np.mean(data3d, axis=1)
        else:
            data2d = data3d[:,slice,:]

        fsize  = fontsize
        margins= [0, 0.18, 1, 0.92]

    if cut == 'xy':
        xaxis = x
        yaxis = y
        xlabel = r'$x/H_g$'
        ylabel = r'$y/H_g$'
        if slice == None:
            data2d = np.mean(data3d, axis=2)
        else:
            data2d = data3d[:,:,slice]

        fsize  = fontsize*1.6
        margins= [0.15, 0., 0.77, 1]

    if cut == 'yz':
        xaxis = y
        yaxis = z
        xlabel = r'$y/H_g$'
        ylabel = r'$z/H_g$'
        if slice == None:
            data2d = np.mean(data3d, axis=0)
        else:
            data2d = data3d[slice,:,:]

        fsize  = fontsize
        margins= [0, 0.18, 1, 0.92]

    if arange == None:
        xmin, xmax = -np.amax(xaxis), np.amax(xaxis)
        ymin, ymax = -np.amax(yaxis), np.amax(yaxis)
    else:
        xmin, xmax = arange[0], arange[1]
        ymin, ymax = arange[2], arange[3]

    if aspect == None:
        aspect = (ymax-ymin)/(xmax-xmin)

    if log != None:
        data2d = np.log10(data2d)

    if plotrange == None:
        x_mask = (xaxis >= xmin) & (xaxis <= xmax)
        y_mask = (yaxis >= ymin) & (yaxis <= ymax)
        data_within_range = data2d[np.ix_(x_mask, y_mask)]

        minv = np.amin(data_within_range)
        maxv = np.amax(data_within_range)
    else:
        minv = plotrange[0]
        maxv = plotrange[1]

    levels = np.linspace(minv, maxv, nlev)
    clevels = np.linspace(minv, maxv, nclev)

    plt.rc('font', size=fsize, weight='bold')

    # Choose a baseline width for the figure
    baseline_width = 10  # in inches

    # Calculate figsize dynamically
    fig_width   = baseline_width
    fig_height  = fig_width * aspect

    # Adjust for the color bar (add 15% width)
    colorbar_space_ratio = 0.15
    fig_width += fig_width * colorbar_space_ratio

    figsize     = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)#, constrained_layout=True)

    ax.set_box_aspect(aspect)

    fig.subplots_adjust(left=margins[0], bottom=margins[1], right=margins[2], top=margins[3])

    cp = plt.contourf(xaxis, yaxis, np.transpose(data2d), levels, cmap=cmap)

    pos         = ax.get_position()
    cbar_width  = 0.02  # Width of the color bar as a fraction of the figure width
    cbar_x      = pos.x1*1.01  # Slight padding to the right of the plot
    cbar_y      = pos.y0  # Align bottom of color bar with bottom of plot
    cbar_height = pos.height  # Match color bar height to plot height
    cbar_ax     = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
    cbar        = fig.colorbar(cp, cax=cbar_ax, ticks=clevels, label=clabel, format='%.2f')
    # cbar        = plt.colorbar(cp, ax=ax, ticks=clevels, format='%.2f')

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.xaxis.set_major_locator(plt.MaxNLocator(1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title != None:
        ax.set_title(title, weight='bold')
    else:
        ax.set_title(r't={0:3.0f}'.format(tstamp)+"P", weight='bold')

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

def load_module_from_path(module_name, file_path):
    """
    Dynamically load a module given its name and file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    prefix = '/Users/minkailin/Projects/idefix-mkl/runs/'
    # cases  = ['fid_2D','gap_2D']
    # cases  = ['gap_2D','gap_2D_UR']
    # cases  = ['fid_3D_damping']
    # cases = ['fid_2D_Nr0.01']
    # cases  = ['fid_3D_damping_HR']
    # cases = ['fid_3D_Nr0.01_damping']
    # cases = ['fid_3D_Nr1e-3_damping_HR']
    # cases = ['fid_3D_gap_Nr0.01']
    cases = ['fid_3D_gap_Nr0.05']
    locs   = [prefix + s for s in cases]
    # locs=['.']
    params = load_module_from_path("COS_linear_nonaxi", '/Users/minkailin/Projects/idefix-mkl/setups/COS-GPU/linear/COS_linear_nonaxi.py')
    ini_file_path = os.path.join(locs[0], "idefix.ini")
    params.load_ini_file(ini_file_path)

    Nx = 128
    x_vals = np.linspace(params.x_min, params.x_max, Nx)  # Example x range from 0 to Lx
    theta0, eigens = params.find_theta0_and_s(x_vals)
    sgrow          = eigens.real

    PlotMaxEvol1D(locs[0], tinterval=(50, 100), var='rmsx', logscale=True, gtheory=sgrow)
    # Plot1DProfile(locs[0]+'/', 4, 'rho', profile='x', plotrange=None, pert=False, title=None)

    # prefix = '/Users/minkailin/Projects/idefix-mkl/runs/'
    # cases  = ['fid_2D']
    # locs   = [prefix + s for s in cases]
    # labels = ['Idefix', 'Idefix (gap)']
    labels = ['Idefix']
    # labels = ['Idefix 3D']

    # PlotMaxEvol1DCompare(locs, labels, var='rmsx', xrange=[0,1000], yrange=[1e-6,2],
    #                        logscale=True)

    # #Plot1DBuoyancy(locs[0]+'/', 0)
    # # PlotMaxEvol1D(locs[0], tinterval=(0, 10), var='vg', logscale=True, gtheory=sgrow)

    prefix = '/Users/minkailin/Projects/SSP24/runs/'
    # dcases  = ['fid_2D', 'gap_2D']
    # dcases  = ['gap_2D', 'gap_2D_UR']
    dcases  = ['fid_3D_damping_Re5e4']
    # dcases  = ['fid_3D_gap_Nr0.01']
    dlocs  = [prefix + s for s in dcases]
    # dlabels= ['Dedalus', 'Dedalus (gap)']
    # dlabels= ['Dedalus 3D']
    dlabels= ['Dedalus (Re=5e4)']
    # PlotMaxEvol1DCompare(locs, labels, dlocs=dlocs, dlabels=dlabels, var='rmsx', xrange=[0,1000], yrange=[1e-6,2],
    #                        logscale=True)
    # PlotMaxEvol1DCompare(locs, labels, dlocs=dlocs, dlabels=dlabels, var='amf', xrange=[0,1000], yrange=[1e-9,1e-1],
    #                         logscale=True,avg=100)

    for n in range(1,6):
        Plot2DContour(locs[0]+'/', n, 'vx', cut='xz', pert=False)
        Plot2DContour(locs[0]+'/', n, 'vz', cut='xz', pert=False)
        Plot2DContour(locs[0]+'/', n, 'vort', cut='xy', pert=False, nonaxi=True)
        Plot2DContour(locs[0]+'/', n, 'rho', cut='xy', pert=False, nonaxi=True)

    # for n in range(1,101):
    #     Plot2DContour(locs[0]+'/', n, 'vx', cut='xz', pert=False)
    #     Plot2DContour(locs[0]+'/', n, 'vx', cut='xy', pert=False)
    #     Plot2DContour(locs[0]+'/', n, 'vort', cut='xy', pert=False)
    #Plot2DContour('/Users/minkailin/Projects/idefix-mkl/runs/cos3D-damp/', n, 'vort', cut='xy', pert=False)

    #PlotCompareWithDedalus(loc, var='amf', avg=100)

    #Plot1DProfile(loc, k, 'rho', pert=False)
    #Plot1DProfile(loc, k, 'pressure', pert=False)
    #Plot1DProfile('', n, 'vx', pert=False)
    #Plot1DBuoyancy(loc, k)

    # beg = 1
    # end = 8
    # for m, loc in enumerate(locs):
    #     for n in range(beg, end+1):
    #         Plot2DContour(loc+'/', n, 'vx', cut='xz', pert=False)
    #         # Plot2DContour(loc+'/', n, 'vy', cut='xz', pert=False)
    #         Plot2DContour(loc+'/', n, 'vz', cut='xz', pert=False)

    #         Plot2DContour(loc+'/', n, 'vort', cut='xy', pert=False)
