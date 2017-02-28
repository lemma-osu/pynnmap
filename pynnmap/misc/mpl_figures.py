import numpy as np
import pylab as pl

from pynnmap.misc import statistics

# output_type enumeration
(SCREEN, FILE) = range(2)


def draw_scatterplot(x, y, metadata, output_type=SCREEN, output_file=None,
    **kwargs):

    # Unpack the metadata information
    variable = metadata.field_name
    short_desc = metadata.short_description
    units = metadata.units

    # Set up the output figure
    if output_type == SCREEN:
        pl.ion()
    else:
        pl.ioff()

    pl.clf()
    pl.gcf().set_figwidth(3.4)
    pl.gcf().set_figheight(3.0)
    pl.gcf().set_dpi(250)

    # Find the min and max of both axes
    if (x.min() < y.min()):
        abs_min = x.min()
    else:
        abs_min = y.min()

    if (x.max() > y.max()):
        abs_max = x.max()
    else:
        abs_max = y.max()

    # Draw the scatterplot data and title
    pl.scatter(x, y, s=2, c='b', edgecolor='k', linewidth=0.25, **kwargs)
    pl.title(variable + ' : ' + short_desc, size=4.5)

    # Calculate correlation coefficient, normalized RMSE and r_square
    this_corr = statistics.pearson_r(x, y)
    this_rmse = statistics.rmse(x, y) / x.mean()
    this_r2 = statistics.r2(x, y)

    # Draw the annotation text on the figure
    pl.text(0.89, 0.93,
        '1:1', transform=pl.gca().transAxes, size=4.5, rotation=45)
    pl.text(0.05, 0.93,
        'Correlation coefficient:  %.4f' % (this_corr),
        transform=pl.gca().transAxes, size=4.5)
    pl.text(0.05, 0.89,
        'Normalized RMSE: %.4f' % (this_rmse),
        transform=pl.gca().transAxes, size=4.5)
    pl.text(0.05, 0.85,
        'R-square: %.4f' % (this_r2),
        transform=pl.gca().transAxes, size=4.5)

    # Draw the 1:1 line and format the x and y axes
    pl.plot([abs_min, abs_max], [abs_min, abs_max], 'k-', linewidth=0.5)
    ylabel_str = 'Predicted ' + variable
    xlabel_str = 'Observed ' + variable
    if units != 'none':
        ylabel_str += ' (' + units + ')'
        xlabel_str += ' (' + units + ')'
    pl.ylabel(ylabel_str, size=4.5)
    pl.xlabel(xlabel_str, size=4.5)

    import matplotlib.ticker as ticker
    f = ticker.OldScalarFormatter()
    # f.set_powerlimits((-3, 4))
    pl.gca().xaxis.set_major_formatter(f)
    pl.gca().xaxis.set_minor_formatter(f)
    pl.gca().yaxis.set_major_formatter(f)
    pl.gca().yaxis.set_minor_formatter(f)

    pl.xticks(size=4)
    pl.yticks(size=4)

    range = abs_max - abs_min
    pl.xlim(abs_min - (0.01 * range), abs_max + (0.01 * range))
    pl.ylim(abs_min - (0.01 * range), abs_max + (0.01 * range))

    # Position the main axis within the figure
    frame_x = 0.125
    frame_width = 0.855
    frame_y = 0.100
    frame_height = 0.830
    pl.gca().set_position([frame_x, frame_y, frame_width, frame_height])
    pl.gca().axesPatch.set_linewidth(0.2)
    axis = pl.gca()
    for spine in axis.spines:
        axis.spines[spine].set_linewidth(0.2)

    # Set fill and edge for the figure
    pl.gcf().figurePatch.set_edgecolor('k')
    pl.gcf().figurePatch.set_linewidth(2.0)

    # Draw and output to file if requested
    pl.draw()
    if output_type == FILE:
        pl.savefig(output_file, dpi=250, edgecolor='k')


def draw_histogram(histograms, bin_names, metadata, output_type=SCREEN,
                   output_file=None, **kwargs):

    """
    Given one or more series of data with identical edges, draw a comparative
    histogram.

    Parameters
    ----------
    histograms : list of numpy arrays
        Bin counts of all series.  These should be the same length and
        represent the same bin endpoints

    bin_names : list
        Names corresponding to each variable's bins

    metadata : XMLAttributeField instance
        Instance which holds information about variable attributes

    output_type: enumeration (int)
        A valid enumeration value of either SCREEN (0) or FILE (1).  When
        importing this module, access to these enumeration values are
        available at the module level.  If set to FILE, the optional
        parameter output_file should be set as well. Defaults to SCREEN.

    output_file: string
        A string representing a valid file location to which the output
        figure will be written. Defaults to None.

    **kwargs: keyword arguments
        Additional keyword arguments that get passed through to the
        pylab.bar command for specific graphic formatting.  See help on
        pylab.bar for additional documentation.

    Returns
    -------
    None
    """

    # Unpack the metadata information
    variable = metadata.field_name
    short_desc = metadata.short_description
    units = metadata.units

    # Set up the output figure
    if output_type == SCREEN:
        pl.ion()
    else:
        pl.ioff()

    pl.clf()
    pl.gcf().set_figwidth(3.4)
    pl.gcf().set_figheight(3.0)
    pl.gcf().set_dpi(250)

    # Set colors for the different series
    colors = ['r', 'y', 'b', 'm', 'c', 'k', 'g', 'w']

    # Set width of the bars
    width = 0.35

    # Get the number of bins to draw
    ind = np.arange(len(histograms[0]))

    # Draw all these plots
    plots = []
    for i in range(len(histograms)):
        plots.append(
            pl.bar(
                ind + (width * i),
                histograms[i],
                color=colors[i],
                width=width,
                lw=0.2,
                **kwargs))

    # Set up the axes to be used
    pl.title(variable + ' : ' + short_desc, size=4.5)
    pl.ylabel('Area (ha)', size=4.5)
    if units != 'none':
        pl.xlabel(variable + ' (' + units + ')', size=4.5)
    else:
        pl.xlabel(variable, size=4.5)
    pl.gca().ticklabel_format(scilimits=(-10, 10))
    pl.xticks(ind + width, bin_names, size=4.0)
    pl.yticks(size=4.0)
    pl.xlim(0, ind[-1] + 1)

    # Draw the legend
    from matplotlib.font_manager import FontProperties
    pl.legend((plots[0][0], plots[1][0]), ('Plots', 'GNN'),
        prop=FontProperties(size=4.5), borderpad=0.6, loc=(0.75, 0.87))
    pl.gca().get_legend().get_frame().set_linewidth(0.2)

    # Set the height of the y-axis
    global_max = 0
    for hist in histograms:
        local_max = hist.max()
        if local_max > global_max:
            global_max = local_max
    pl.ylim(0, global_max * 1.20)

    # Set the horizontal components of the axes
    frame_x = 0.15
    frame_width = 0.83

    # Determine whether we need to tilt the labels
    # Space allocated to each label
    label_width = frame_width / len(histograms[0])

    # Find the longest label
    max_label = len(bin_names[0])
    for i in range(1, len(bin_names)):
        if len(bin_names[i]) > max_label:
            max_label = len(bin_names[i])

    min_ratio = 0.0094
    rotation = 0.0
    if (label_width / max_label) < min_ratio:
        labels = pl.gca().get_xticklabels()

        # Set the rotation based on this ratio
        m = 60.0 / min_ratio
        x = min_ratio - (label_width / max_label)
        y = m * x
        rotation = 30.0 + y
        pl.setp(labels, 'rotation', rotation)
    else:
        rotation = 0.0

    # Set the vertical components of the axes based on the rotation angle
    frame_y = 0.10
    frame_height = 0.83

    # Adjustment factor is based on rotation angle and maxLabel
    adj_factor = 0.00014 * (rotation * max_label)
    frame_y += adj_factor
    frame_height -= adj_factor

    pl.gca().set_position([frame_x, frame_y, frame_width, frame_height])
    pl.gca().axesPatch.set_linewidth(0.2)
    axis = pl.gca()
    for spine in axis.spines:
        axis.spines[spine].set_linewidth(0.2)

    pl.gcf().figurePatch.set_edgecolor('k')
    pl.gcf().figurePatch.set_linewidth(2.0)

    if output_type == FILE:
        pl.savefig(output_file, dpi=250, edgecolor='k')
