#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         AircraftOperation & Systems, Air Transport Department, ENAC
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.cm as cm

import pickle

from gam_jax.utils import unit

colors = {"general": "green",
          "commuter": "gold",
          "regional": "darkorange",
          "business": "blue",
          "short_medium": "darkviolet",
          "long_range": "red"}

markers = {"piston" : "x",
           "turboprop" : "^",
           "turbofan" : "o"}

#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis functions
#
#-----------------------------------------------------------------------------------------------------------------------

# Set font size
plt.rc('axes',labelsize=12,titlesize=20)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('legend',fontsize=12)


def draw_grid(file_name, data_grid, title):
    """Draw the figure of the flight grid and store it in a file

    :param file_name: file to store the figure
    :param data_grid: data source
    :return:
    """

    fig, ax = plt.subplots()

    table = np.array(data_grid[0])
    capa_list = np.array(data_grid[1])
    range_list = np.array(data_grid[2]) * 1.e-3

    max_val = table.max()
    min_val = table.min() + 0.1

    im = ax.pcolormesh(table,
                       edgecolors='b',
                       linewidth=0.01,
                       cmap=plt.cm.get_cmap("rainbow", 10),
                       norm=plt_colors.LogNorm(vmin=min_val, vmax=max_val))

    # im = ax.pcolormesh(table,
    #                    edgecolors='b',
    #                    linewidth=0.01,
    #                    cmap=plt.cm.get_cmap("rainbow",10),
    #                    norm=colors.Normalize(vmin=min_val, vmax=max_val))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('Ranges (km)', fontsize=14)
    ax.set_ylabel('Seat capacity', fontsize=14)
    ax.xaxis.set_ticks(range(len(range_list)))
    ax.xaxis.set_ticklabels(range_list, fontsize=8, rotation='vertical')
    ax.yaxis.set_ticks(range(len(capa_list)))
    ax.yaxis.set_ticklabels(capa_list, fontsize=8)
    plt.title(title, fontsize=16)
    plt.grid(True)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', aspect=40.)
    plt.tight_layout()
    plt.savefig(file_name, dpi=500, bbox_inches='tight')
    plt.show()


def store_data_to_file(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
        return


def load_data_from_file(file_name):
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError :
        print("Cannot find file : ", file_name)
        data = None
    return data


def read_db(file):
    """Read data base and convert to standard units
    WARNING: special treatment for cruise_speed and max_speed which can be Mach number
    """
    raw_data = pd.read_excel(file)     # Load data base as a Pandas data frame
    un = raw_data.iloc[0:2,0:]                          # Take unit structure only
    df = raw_data.iloc[2:,0:].reset_index(drop=True)    # Remove unit rows and reset index

    for name in df.columns:
        if un.loc[0,name] not in ["string","int"] and un.loc[1,name] != "mach":
            df[name] = unit.convert_from(un.loc[0,name], list(df[name]))
        if un.loc[0,name]=="string":
            df[name] = [str(s) for s in list(df[name])]
        if un.loc[1,name] == "mach":
            for j in df.index:
                if df.loc[j, name] > 1.:
                    df.loc[j, name] = float(unit.convert_from(un.loc[0, name], df.loc[j, name]))
    return df,un


def lin_lst_reg(df, abs, ord, order):
    """Linear least square regression of "ord" versus "abs" with given order
    order is the list of exponents to apply
    """
    def make_mat(param,order):
        mat_list = []
        for j in order:
            mat_list.append(param**j)
        mat = np.vstack(mat_list)
        return mat.T      # Need to transpose the stacked matrix

    param = np.array(list(df[abs]))
    A = make_mat(param, order)
    B = np.array(list(df[ord]))
    (C, res, rnk, s) = np.linalg.lstsq(A, B, rcond=None)

    AC = np.dot(A,C)
    res = np.sqrt(np.sum((AC-B)**2))

    x_reg = np.array(np.linspace(0, max(df[abs]), 400))
    F = make_mat(x_reg, order)
    y_reg = np.dot(F,C)

    return {"coef":C, "res":res, "reg":[x_reg,y_reg]}

def plot_without_reg(df, un, abs, ord, markers=None,color_by="nominal_range", colors=None, leg_loc="lower right",file_name=None):
    """Plot figure with colormap and markers cloud OR with only categories of color cloud """
    fig, axes = plt.subplots(1, 1)

    xrange = [0, unit.convert_to(un.loc[0, abs], max(df[abs]) * 1.05)]
    yrange = [0, unit.convert_to(un.loc[0, ord], max(df[ord]) * 1.05)]

    if (markers is not None) and (colors is not None):
        raise ValueError("Only markers or colors should be given")
    elif colors is not None:
        draw_colored_cloud_on_axis(axes, df, un, abs, ord, colors, xrange=xrange, yrange=yrange)
    elif markers is not None:
        clouds = draw_markers_and_colormap_cloud_on_axis(axes, df, un, abs, ord,markers,color_by=color_by, xrange=xrange, yrange=yrange)
        fig.colorbar(clouds[0], orientation='vertical',label=color_by+"("+un.loc[0,color_by]+")")
    else:
        raise ValueError("You must specify colors or markers.")

    plt.legend(loc=leg_loc)
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()
    return


def draw_reg(df, un, abs, ord, reg, colors, title=None, leg_loc="lower right", file_name=None):
    """Draw the cloud of point in perspective of the regression given into "reg" as [abs_list, ord_list]
    Coloration of each airplane type is given into "colors"
    """
    fig,axes = plt.subplots(1,1)
    fig.canvas.manager.set_window_title("Regression")

    if title is None:
        title = ord + " - " + abs
    fig.suptitle(title, fontsize=16)

    xrange=[0, unit.convert_to(un.loc[0,abs],max(df[abs])*1.05)]
    yrange=[0, unit.convert_to(un.loc[0,ord],max(df[ord])*1.05)]
    draw_colored_cloud_on_axis(axes, df, un, abs, ord, colors, xrange=xrange, yrange=yrange)

    if len(reg[0])>0:
        plt.plot(unit.convert_to(un.loc[0,abs],reg[0]), unit.convert_to(un.loc[0,ord],reg[1]), linewidth=2, color="grey")

    plt.legend(loc=leg_loc)
    # plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    plt.show()



def subplots_by_varname(df,un,var_names,figwidth=12,savefig=False):
    """create set of subplots for each variable in the list var_name"""

    if len(var_names)<1:                          # Check the length of var_names
        raise ValueError("var_names is empty")

    figsize = (figwidth, figwidth*0.4*len(var_names) )
    fig,axes = plt.subplots(len(var_names),2,figsize=figsize)  # Create subplots and eventually refactor the axis list
    if len(var_names)==1:
        axes = [axes]

    first_line=True
    for (line,var) in zip(axes,var_names):                     # fill the subplots
        if first_line:                                                 # add title to first line
            first_line=False
            line[0].set_title(r'Bissectrice $x_{mod}=f(x)$')
            line[1].set_title(r'Erreur relative $\frac{x_{mod}-x}{x}$ (%)')

        var_range = [0, unit.convert_to(un.loc[0, var], max(df[var]))]

        # first cell  : bisectrice plot
        line[0].plot(var_range, var_range, '-k', lw=2)  # draw y=x line
        draw_colored_cloud_on_axis(line[0], df, un, var, 'model_'+var, xrange=var_range)      # draw x_model versus x_data

        # second cell : relative error
        line[1].plot(var_range, [0, 0], '-k', lw=2)
        df['error_'+var] = (df['model_'+var]-df[var])/df[var]*100
        un['error_'+var] = 'no_dim'
        draw_colored_cloud_on_axis(line[1], df, un, var, 'error_'+var, xrange=var_range, yrange=[-100,100])

    plt.tight_layout()
    if savefig:
        plt.savefig("multiplot.pdf")
    plt.show()


def draw_colored_cloud_on_axis(ax, df, un, abs, ord, colors, xrange=None, yrange=None, leg_loc="lower right"):
    """Build a colored scatter plot according to given colors categories.
    :param ax: a figure axis to plot on
    :param df: a dataframe
    :param un: the dict of units to use for the axis scale
    :param abs: the label of the dataframe column to plot on x-axis
    :param ord: the label of the dataframe column to plot on y-axis
    :param colors: a dictionary with label of the category as a key, and the color of the category as value. {'turboprop':'red'}
    :param xrange: list of 2 value for the x-axis range. Matplotlib default if not specified.
    :param yrange: list of 2 value for the y-axis range. Matplotlib default if not specified.
    :return:
    """
    cloud = []
    for typ in colors.keys():
        abs_list = unit.convert_to(un.loc[0, abs], list(df.loc[df['airplane_type'] == typ][abs]))
        ord_list = unit.convert_to(un.loc[0, ord], list(df.loc[df['airplane_type'] == typ][ord]))
        if len(abs_list)>0:
            subcloud = ax.scatter(abs_list, ord_list, marker="o", c=colors[typ], s=10, label=typ)
            cloud.append(subcloud)
    ax.set_ylabel(ord + ' (' + un.loc[0, ord] + ')', fontsize=14)
    ax.set_xlabel(abs + ' (' + un.loc[0, abs] + ')', fontsize=14)
    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.legend(handles=cloud, loc=leg_loc)
    ax.grid(True)

def draw_markers_and_colormap_cloud_on_axis(ax, df, un, abs, ord, markers,color_by="nominal_range", xrange=None, yrange=None, leg_loc="upper left"):
    """Build a colored scatter plot according to a given color scale, with a colorbar on the side.
    :param ax: a figure axis to plot on
    :param df: a dataframe
    :param un: the dict of units to use for the axis scale
    :param abs: the label of the dataframe column to plot on x-axis
    :param ord: the label of the dataframe column to plot on y-axis
    :param markers: a dictionary with label of the category as a key, and the matplotlib marker as value.
    :param color_by: the label of the dataframe column for the color mapping
    :param xrange: list of 2 value for the x-axis range. Matplotlib default if not specified.
    :param yrange: list of 2 value for the y-axis range. Matplotlib default if not specified.
    :return:
    """
    cloud = []
    # Add a color list to the dataframe
    vmin = unit.convert_to(un.loc[0, color_by],df[color_by].min())
    vmax = unit.convert_to(un.loc[0, color_by],df[color_by].max())

    for typ in markers.keys():
        abs_list = unit.convert_to(un.loc[0, abs], list(df.loc[df['engine_type'] == typ][abs]))
        ord_list = unit.convert_to(un.loc[0, ord], list(df.loc[df['engine_type'] == typ][ord]))
        color_list = unit.convert_to(un.loc[0, color_by], list(df.loc[df['engine_type'] == typ][color_by]))
        if len(abs_list)>0:
            subcloud = ax.scatter(abs_list, ord_list, c=color_list, vmin=vmin,vmax=vmax, marker=markers[typ], s=25, label=typ)
            cloud.append(subcloud)
    ax.set_ylabel(ord + ' (' + un.loc[0, ord] + ')', fontsize=14)
    ax.set_xlabel(abs + ' (' + un.loc[0, abs] + ')', fontsize=14)
    # ax.set_yscale('log')
    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.legend(handles=cloud, loc=leg_loc)
    ax.grid(True)
    return cloud


def get_error(df, un, abs, ord, reg, abs_interval):

    df1 = df[abs_interval[0]<=df[abs]].reset_index(drop=True).copy()
    df1 = df1[df1[abs]<=abs_interval[1]].reset_index(drop=True)

    fct = interp1d(reg[0], reg[1], kind="cubic", fill_value='extrapolate')

    df1['relative_error'] = (fct(df1[abs]) - df1[ord]) / df1[ord]

    print("Mean relative error = ", np.mean(list(df1['relative_error'])))
    print("Variance of relative error = ", np.var(list(df1['relative_error'])))

    draw_hist(list(df1['relative_error']), "error")


def draw_hist(rer,title):
    """Draw the histogram of relative errors given into "reg" as [abs_list, ord_list]
    """
    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title("Relative error distribution")
    fig.suptitle(title, fontsize=12)

    plt.hist(rer, bins=20, range=(-1,1))

    plt.ylabel('Count')
    plt.xlabel('Relative Error')
    plt.show()


def do_regression(df, un, abs, ord, colors, order, title=None):
    """Perform regression and draw the corresponding graph
    """
    dict = lin_lst_reg(df, abs, ord, order)
    print("Coef = ", dict["coef"])
    print("Res = ", dict["res"])
    draw_reg(df, un, abs, ord, dict["reg"], colors, title)
    return dict




#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis
#
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    # path_to_data_base = "All_Data_extract.xlsx"
    path_to_data_base = "../database/CADO_airplane_database_v1.0.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

    #----------------------------------------------------------------------------------
    abs = "max_fuel/mtow"
    ord = "mtow"
    color_by = "nominal_range"

    df1 = df[df['airplane_type'] != 'business'].reset_index(drop=True).copy()
    df1[abs] = df["max_fuel"]/df["mtow"]
    un1 = un.copy()
    un1[abs] = "no_dim"

    plot_without_reg(df1, un1, abs, ord,markers=markers,color_by="nominal_range")

    exit()

    #----------------------------------------------------------------------------------
    abs = "n_pax*nominal_range"                           # Name of the new column
    ord = "mtow"

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    # df1 = df1[df1['MTOW']<60000].reset_index(drop=True).copy()
    un1 = un.copy()

    df1[abs] = df1['n_pax']*df1['nominal_range']**0.75    # Add the new column to the dataframe
    un1[abs] = "km"                 # Add its unit

    order = [0.85]
    dict = do_regression(df1, un1, abs, ord, colors, order)


    #----------------------------------------------------------------------------------
    abs = "n_pax*nominal_range"                           # Name of the new column
    ord = "mtow"

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    df1 = df1[df1['mtow']<100000].reset_index(drop=True).copy()
    un1 = un.copy()

    df1[abs] = df1['n_pax']*df1['nominal_range']**0.25    # Add the new column to the dataframe
    un1[abs] = "km"                 # Add its unit

    order = [1]
    dict = do_regression(df1, un1, abs, ord, colors, order)


    # perform regressions
    #-------------------------------------------------------------------------------------------------------------------
    abs = "mtow"
    ord = "owe"

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    # df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t

    # order = [1]
    order = [2, 1]
    dict_owe = do_regression(df, un, abs, ord, colors, order)


    #----------------------------------------------------------------------------------
    abs = "mtow"
    ord = "total_power"                           # Name of the new column

    df[ord] = df['max_power']*df['n_engine']      # Add the new column to the dataframe
    un[ord] = un['max_power']                     # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1]
    dict = do_regression(df, un, abs, ord, colors, order)


    #----------------------------------------------------------------------------------
    abs = "(mtow/wing_area)**0.5"
    ord = "approach_speed"

    df[abs] = (df['mtow'] / df['wing_area'])**0.5   # Add the new column to the dataframe
    un[abs] = "kg/m2"                                      # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [1]
    dict = do_regression(df, un, abs, ord, colors, order)


    #----------------------------------------------------------------------------------
    abs = "nominal_range"                           # Name of the new column
    ord = "n_pax"

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    un1 = un.copy()

    dict = draw_reg(df1, un1, abs, ord, [[],[]], colors)


    #----------------------------------------------------------------------------------
    abs = "nominal_range"                           # Name of the new column
    ord = "PKoM"


    df1[ord] = df1['n_pax']*df1['nominal_range']/df1['mtow']     # Add the new column to the dataframe
    un1[ord] = "m/kg"                 # Add its unit

    # order = [1.8, 0.8]
    dict = draw_reg(df1, un1, abs, ord, [[],[]], colors)

