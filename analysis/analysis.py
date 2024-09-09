import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Patch
import pandas as pd
import sys
import toml

sys.path.append('/Users/mueller/ROOT/install/lib/')
from ROOT import TEfficiency # type: ignore

def prepare_config(variables, plots, tables, input_files, systematics):
    """
    Reads and assembles the various configuration files needed for the
    analysis.

    Parameters
    ----------
    variables: str
        The path to the variables configuration file.
    plots: str
        The path to the plots configuration file.
    tables: str
        The path to the tables configuration file.
    input_files: str
        The path to the input files configuration file.
    systematics: str
        The path to the systematics configuration file.
    
    Returns
    -------
    cfg: dict
        The dictionary containing the configuration information.
    """
    cfg = toml.load(systematics)
    cfg.update(toml.load(variables))
    cfg.update(toml.load(plots))
    cfg.update(toml.load(input_files))
    cfg.update(toml.load(tables))
    return cfg

def read_log(path, tag, header, category_selector=None):
    """
    Reads an input log file and extracts lines with the specified tag
    into a Pandas DataFrame.

    Parameters
    ----------
    path: str
        The full path of the input log file.
    tag: str
        The identifier that tags relevant lines in the log file.
    header: list[str]
        The list of column names for the CSV file.
    category_selector: callable
        A function that takes a line and returns True if the line is to
        be included in the output DataFrame.

    Returns
    -------
    data: Pandas.DataFrame
        The DataFrame containing the requested information.
    """
    input_file = open(path)
    lines = input_file.readlines()
    selected = [x.strip('\n').split(',')[1:] for x in lines if tag in x]
    selected = [x if x[-1] != '' else x[:-1] for x in selected]
    data = pd.DataFrame(selected, columns=header[:len(selected[0])])
    for k in header[:len(data.columns)]:
        data[k] = pd.to_numeric(data[k], errors='coerce', downcast='float')
        if data[k].apply(float.is_integer).all():
            data[k] = data[k].astype(int)
    if category_selector is not None:
        data = data[data.apply(category_selector, axis=1)]
    return data

def prepare_data(cfg, cryo=None):
    """
    Loads the input datasets and covariance matrices and packages them
    in a dictionary.

    Parameters
    ----------
    cfg: dict
        The configuration dictionary for the plot.
    
    Returns
    -------
    data: dict
        The dictionary containing the input datasets and covariance
        matrices.
    cryo: int
        The cryostat to select for the analysis.
    """
    header = [k for k,v in cfg['header'].items()]
    header_data = [k for k,v in cfg['header'].items() if v == 'reco']

    if cryo is not None:
        cryo = (cryo + 1 ) % 2
    else:
        cryo = 3

    data = dict()
    data['signal_1mu1p'] = read_log(cfg['input_files']['simulation'], 'SIGNAL', header, lambda x: ((x['category'] == 0) and (np.abs(x['trigger'] - 1500) < 10) and (x['cryostat'] != cryo)))
    data['signal_1muNp'] = read_log(cfg['input_files']['simulation'], 'SIGNAL', header, lambda x: ((x['category'] == 0 or x['category'] == 2) and (np.abs(x['trigger'] - 1500) < 10) and (x['cryostat'] != cryo)))
    data['signal_1muX'] = read_log(cfg['input_files']['simulation'], 'SIGNAL', header, lambda x: ((x['category'] == 0 or x['category'] == 2 or x['category'] == 4) and (np.abs(x['trigger'] - 1500) < 10) and (x['cryostat'] != cryo)))
    data['selected_1mu1p'] = read_log(cfg['input_files']['simulation'], 'SELECTED', header, lambda x: ((x['selected_1mu1p'] == 1) & (x['crtpmt_match'] == 1) & (np.abs(x['trigger'] - 1500) < 10) and (x['cryostat'] != cryo)))
    data['selected_1muNp'] = read_log(cfg['input_files']['simulation'], 'SELECTED', header, lambda x: ((x['selected_1muNp'] == 1) & (x['crtpmt_match'] == 1) & (np.abs(x['trigger'] - 1500) < 10) and (x['cryostat'] != cryo)))
    data['selected_1muX'] = read_log(cfg['input_files']['simulation'], 'SELECTED', header, lambda x: ((x['selected_1muX'] == 1) & (x['crtpmt_match'] == 1) & (np.abs(x['trigger'] - 1500) < 10) and (x['cryostat'] != cryo)))
    data['data_1mu1p'] = read_log(cfg['input_files']['data'], 'DATA', header_data, lambda x: ((x['selected_1mu1p'] == 1) & (x['crtpmt_match'] == 1) and (x['cryostat'] != cryo)))
    data['data_1muNp'] = read_log(cfg['input_files']['data'], 'DATA', header_data, lambda x: ((x['selected_1muNp'] == 1) & (x['crtpmt_match'] == 1) and (x['cryostat'] != cryo)))
    data['data_1muX'] = read_log(cfg['input_files']['data'], 'DATA', header_data, lambda x: ((x['selected_1muX'] == 1) & (x['crtpmt_match'] == 1) and (x['cryostat'] != cryo)))

    data['offbeam_1mu1p'] = read_log(cfg['input_files']['offbeam'], 'DATA', header_data, lambda x: ((x['selected_1mu1p'] == 1) and (x['cryostat'] != cryo)))
    data['offbeam_1muNp'] = read_log(cfg['input_files']['offbeam'], 'DATA', header_data, lambda x: ((x['selected_1muNp'] == 1) and (x['cryostat'] != cryo)))
    data['offbeam_1muX'] = read_log(cfg['input_files']['offbeam'], 'DATA', header_data, lambda x: ((x['selected_1muX'] == 1) and (x['cryostat'] != cryo)))

    matches = pd.read_csv(cfg['input_files']['offbeam_matches'])
    
    for channel in ['1mu1p', '1muNp', '1muX']:
        res = []
        for run, event in zip(data[f'offbeam_{channel}']['run'], data[f'offbeam_{channel}']['event']):
            q = matches[((matches['run'] == run) & (matches['event'] == event))]['crtpmt_match']
            if len(q) != 0:
                res.append(q.iloc[0])
            else:
                res.append(1)
        data[f'offbeam_{channel}']['crtpmt_match'] = res
        data[f'offbeam_{channel}'] = data[f'offbeam_{channel}'].loc[data[f'offbeam_{channel}']['crtpmt_match'] == 1]

    scale = (cfg['exposure']['data'] / cfg['exposure'][cfg['input_files']['detsys']['exposure']])
    data['systematics_1mu1p'] = {k : scale**2 * v if 'ratio' not in k else v for k,v in np.load(cfg['input_files']['detsys']['1mu1p']).items()}
    data['systematics_1muNp'] = {k : scale**2 * v if 'ratio' not in k else v for k,v in np.load(cfg['input_files']['detsys']['1muNp']).items()}
    data['systematics_1muX'] = {k : scale**2 * v if 'ratio' not in k else v for k,v in np.load(cfg['input_files']['detsys']['1muX']).items()}
    scale = (cfg['exposure']['data'] / cfg['exposure'][cfg['input_files']['multisim']['exposure']])
    data['systematics_1mu1p'].update({k : scale**2 * v if 'ratio' not in k else v for k,v in np.load(cfg['input_files']['multisim']['1mu1p']).items()})
    data['systematics_1muNp'].update({k : scale**2 * v if 'ratio' not in k else v for k,v in np.load(cfg['input_files']['multisim']['1muNp']).items()})
    data['systematics_1muX'].update({k : scale**2 * v if 'ratio' not in k else v for k,v in np.load(cfg['input_files']['multisim']['1muX']).items()})

    for var in cfg['variables']:
        bins = cfg['variables'][var]['bins']
        if 'softmax' not in var:
            var = f'reco_{var}'

        scale = (cfg['exposure']['data'] / cfg['exposure'][cfg['input_files']['multisim']['exposure']])
        for c in ['1mu1p', '1muNp', '1muX']:
            content = np.histogram(data[f'selected_{c}'][var], bins=int(bins[0]), range=bins[1:])[0]
            cov = np.outer(np.sqrt(content), np.sqrt(content))
            percent_fudge = (2 / (100*np.sum(np.sqrt(np.diag(cov))) / np.sum(content)))**2
            data[f'systematics_{c}'][f'pot_{var}'] = scale**2 * percent_fudge * cov
            data[f'systematics_{c}'][f'flux_{var}'] += data[f'systematics_{c}'][f'pot_{var}']
            data[f'systematics_{c}'][f'statistical_{var}'] = scale**2 * np.diag(content)
            data[f'systematics_{c}'][f'total_{var}'] = np.sum([data[f'systematics_{c}'][f'{k}_{var}'] for k in ['detector', 'genie', 'flux', 'statistical']], axis=0)

        onbeam_gates = cfg['exposure']['onbeam_events'] / cfg['exposure']['onbeam_rate']
        offbeam_gates = cfg['exposure']['offbeam_events'] / cfg['exposure']['offbeam_rate']
        scale = onbeam_gates / offbeam_gates
        for c in ['1mu1p', '1muNp', '1muX']:
            data[f'systematics_{c}'][f'offbeamstats_{var}'] = scale**2 * np.diag(np.histogram(data[f'offbeam_{c}'][var], bins=int(bins[0]), range=bins[1:])[0])
            data[f'systematics_{c}'][f'total_{var}'] += data[f'systematics_{c}'][f'offbeamstats_{var}']

    rf = uproot.open(cfg['input_files']['simulation_root'])
    offbeam = uproot.open(cfg['input_files']['offbeam_root'])
    for c in ['1mu1p', '1muNp', '1muX']:
        onbeam_gates = cfg['exposure']['onbeam_events'] / cfg['exposure']['onbeam_rate']
        offbeam_gates = cfg['exposure']['offbeam_events'] / cfg['exposure']['offbeam_rate']
        scale = onbeam_gates / offbeam_gates
        offbeam_contamination = scale * offbeam[f'sOffbeam{c}Cut'].values()
        
        cuts = ['NoCut', 'FVCut', 'FVConCut', f'FVConTop{c}Cut', f'All{c}Cut']
        mcscale = cfg['exposure']['data'] / cfg['exposure']['montecarlo']
        d = [np.rint(mcscale * rf[f'sCountPTT_{k}'].values()[:,0]) for k in cuts]
        for i in range(len(d)):
            d[i][7] += np.rint(offbeam_contamination[i])
        d.append(np.rint(mcscale * np.histogram(data[f'selected_{c}']['category'], bins=range(11))[0]))
        d[-1][7] += np.rint(scale * len(data[f'offbeam_{c}']))
        data[f'purity_table_{c}'] = np.stack(d)

        d = [np.rint(mcscale * rf[f'sCountTTP_{k}'].values()[:,0]) for k in cuts]
        selected = ((data[f'signal_{c}'][f'selected_{c}'] == 1) & (data[f'signal_{c}']['crtpmt_match'] == 1))
        d.append(mcscale * np.histogram(data[f'signal_{c}']['category'].loc[selected], bins=range(11))[0])
        data[f'efficiency_table_{c}'] = np.stack(d)

    return data

def add_error_boxes(ax, x, y, xerr, yerr, **kwargs):
    """
    Adds error boxes to the input axis.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis to which the error boxes are to be added.
    x: numpy.array
        The x-coordinates of the error boxes.
    y: numpy.array
        The y-coordinates of the error boxes.
    xerr: numpy.array
        The x-error values of the error boxes.
    yerr: numpy.array
        The y-error values of the error boxes.
    kwargs: dict
        Keyword arguments to be passed to the errorbar function.

    Returns
    -------
    None.
    """
    boxes = [Rectangle((x[i] - xerr[i], y[i] - yerr[i]), 2 * np.abs(xerr[i]), 2 * yerr[i]) for i in range(len(x))]
    pc = PatchCollection(boxes, **kwargs)
    ax.add_collection(pc)
    return boxes[0]

def mark_icarus_preliminary(ax, cfg):
    """
    Mark the plot as ICARUS preliminary.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to mark.
    cfg: dict
        The configuration dictionary for the watermark.
        
    Returns
    -------
    None.
    """
    yrange = ax.get_ylim()
    usey = yrange[1] + 0.025*(yrange[1] - yrange[0])
    xrange = ax.get_xlim()
    usex = xrange[0] + 0.025*(xrange[1] - xrange[0])
    ax.text(x=usex, y=usey, s=fr'{cfg["text"]}', fontsize=14, color='#d67a11')
    
    """
    if release is None and simulation:
        ax.text(x=usex, y=usey, s=r'$\bf{ICARUS}$ Simulation Preliminary', fontsize=14, color='#d67a11')
    elif simulation:
        ax.text(x=usex, y=usey, s=r'$\bf{ICARUS}$ Simulation ' + release, fontsize=14, color='#d67a11')
    elif release is not None:
        ax.text(x=usex, y=usey, s=r'$\bf{ICARUS}$ Preliminary ' + release, fontsize=14, color='#d67a11')
    else:
        ax.text(x=usex, y=usey, s=r'$\bf{ICARUS}$ Preliminary', fontsize=14, color='#d67a11')
    """

def mark_pot(ax, pot):
    """
    Add the POT information to the plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to mark.
    pot: float
        The number of protons on target.

    Returns
    -------
    None.
    """
    yrange = ax.get_ylim()
    usey = yrange[1] + 0.02*(yrange[1] - yrange[0])
    xrange = ax.get_xlim()
    usex = xrange[1] - 0.02*(xrange[1] - xrange[0])
    usepot = pot/1.0e19
    ax.text(x=usex, y=usey, s=f'{usepot:.2f}'+r'$\times 10^{19}$ POT', fontsize=13, color='black', horizontalalignment='right')


def add_stacked_histogram(ax, data, cfg, exposure, offbeam=None):
    """
    Plots a stacked histogram consisting of multiple categories
    for the reconstructed variable of interest.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to plot on.
    data: dict
        The dictionary containing the datasets to plot.
    cfg: dict
        The configuration dictionary for the plot.
    exposure: dict
        The exposure information for the plot.
    offbeam: dict
        The offbeam data for the plot.

    Returns
    -------
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    """

    contents = list()
    centers = list()
    labels = list()
    width = list()
    counts = list()
    for m in cfg['merge'][::-1]:
        mask = np.isin(data[cfg['categorical_var']], m)
        c, e = np.histogram(data[cfg['var']][mask], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
        contents.append((exposure['data'] / exposure['montecarlo']) * c)
        centers.append((e[1:] + e[:-1]) / 2.0)
        width.append(np.diff(e))
        labels.append(cfg['categories'][m[0]])
        counts.append((exposure['data'] / exposure['montecarlo']) * np.sum(mask))

    if offbeam is not None:
        cosmic_index = labels.index('Cosmic')
        c, _ = np.histogram(offbeam[cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
        offbeam_scale = (exposure['onbeam_events'] / exposure['onbeam_rate']) / (exposure['offbeam_events'] / exposure['offbeam_rate'])
        scaled_value = c * offbeam_scale
        contents[cosmic_index] = scaled_value
        counts[cosmic_index] += offbeam_scale * np.sum(len(offbeam[cfg['var']]))

    if cfg.get('normalize', False):
        contents = [c / np.sum(contents) for c in contents]

    colors = [f'C{i}' for i in cfg['colors']][::-1]
    ax.hist(centers, weights=contents, bins=int(cfg['bins'][0]), range=cfg['bins'][1:], label=labels, color=colors, histtype='barstacked')

    h, l = ax.get_legend_handles_labels()
    if 'show_percentage' in cfg.keys() and cfg['show_percentage'] and not cfg.get('normalize', False):
        l = [f'{l} ({counts[li]:.1f}, {counts[li] / np.sum(counts):.01%})'for li, l in enumerate(l)]
    elif cfg.get('normalize', False):
        l = [f'{l} ({counts[li] / np.sum(counts):.01%})'for li, l in enumerate(l)]
    else:
        l = [f'{l} ({counts[li]:.1f})'for li, l in enumerate(l)]
    h = h[::-1]
    l = l[::-1]

    if cfg.get('normalize', False):
        cfg['ylim'][1] = cfg['override_ylim']
    else:
        cfg['ylim'][1] = cfg['ylim_norm'][1] * (int(np.sum(contents) / cfg['ylim_norm'][0] / cfg['ylim_norm'][1])+1)
    ax.legend(h, l, ncol=cfg['legend_ncol'])
    return h, l

def calculate_pureff(bin_assignment, selected, nbins):
    """
    Calculates the purity/efficiency of the selection using the bin assignment
    of each entry and a boolean mask indicating whether the entry was
    selected/signal. Uses the ROOT TEfficiency class.

    Parameters
    ----------
    bin_assignment: numpy.ndarray
        The bin assignment of each entry.
    selected: numpy.ndarray
        The boolean mask indicating whether the entry was selected/signal.
    nbins: int
        The number of bins in the histogram.

    Returns
    -------
    result: numpy.ndarray
        The purity/efficiency of the selection in each bin and the associated
        lower and upper errors.
    """
    res = TEfficiency('res', 'res', int(nbins), 0, nbins)
    for b, s in zip(bin_assignment, selected):
        res.Fill(int(s), b-1)
    res.SetStatisticOption(6)
    return np.array([(res.GetEfficiency(b), res.GetEfficiencyErrorLow(b), res.GetEfficiencyErrorUp(b)) for b in range(1, int(nbins)+1)])

def add_purity_efficiency(ax, data, cfg):
    """
    Plots the purity and efficiency of the selection as a function
    of the variable of interest.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to plot on.
    data: dict
        The dictionary containing the datasets to plot.
    cfg: dict
        The configuration dictionary for the plot.

    Returns
    -------
    None.
    """
    
    bin_edges = np.histogram_bin_edges(np.ones(100), bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
    xerr = np.diff(bin_edges) / 2.0
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    seldefs = {'1mu1p': lambda x: x[f'selected_1mu1p'].to_numpy(), '1muNp': lambda x: x[f'selected_1muNp'].to_numpy(), '1muX': lambda x: x[f'selected_1muX'].to_numpy()}
    sigdefs = {'1mu1p': lambda x: np.isin(x['category'], [0]), '1muNp': lambda x: np.isin(x['category'], [0, 2]), '1muX': lambda x: np.isin(x['category'], [0, 2, 4])}

    sel = data[f'signal_{cfg["channel"]}']
    truth_var = cfg['var'] if 'true' in cfg['var'] else f'true_{cfg["var"][5:]}'
    bin_assignment = np.digitize(sel[truth_var], bin_edges, right=True)
    bin_contents = np.array([np.sum(bin_assignment == i) for i in range(1,len(bin_edges))])
    yerr = np.sqrt(bin_contents)

    efficiency = calculate_pureff(bin_assignment, seldefs[cfg['channel']](sel), cfg['bins'][0])
    mask = efficiency[:,0] != 0.5
    ax.errorbar(bin_centers[mask], efficiency[mask,0], xerr=xerr[mask], yerr=efficiency[mask,1:].transpose(), color='seagreen', label='Efficiency', fmt='o')

    sel = data[f'selected_{cfg["channel"]}']
    bin_assignment = np.digitize(sel[cfg['var']], bin_edges, right=True)
    bin_contents = np.array([np.sum(bin_assignment == i) for i in range(1,len(bin_edges))])
    yerr = np.sqrt(bin_contents)

    efficiency = calculate_pureff(bin_assignment, sigdefs[cfg["channel"]](sel), cfg['bins'][0])
    mask = efficiency[:,0] != 0.5
    ax.errorbar(bin_centers[mask], efficiency[mask,0], xerr=xerr[mask], yerr=efficiency[mask,1:].transpose(), color='rebeccapurple', label='Purity', fmt='o')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction')
    ax.legend()

def add_systematic(ax, data, cfg, exposure, h, l, offbeam=None):
    """
    Adds the systematic error to the plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to plot on.
    data: dict
        The dictionary containing the datasets to plot.
    cfg: dict
        The configuration dictionary for the plot.
    exposure: dict
        The dictionary containing the exposure information.
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    offbeam: pd.DataFrame
        The offbeam data for the plot.

    Returns
    -------
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    """
    systematic, syslabel = cfg['systematic']
    covariance = data[f'systematics_{cfg["channel"]}'][f'{systematic}_{cfg["var"]}']
    
    c, e = np.histogram(data[f'{cfg["type"]}_{cfg["channel"]}'][cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
    centers = (e[1:] + e[:-1]) / 2.0
    width = np.diff(e)
    content = (exposure['data'] / exposure['montecarlo']) * c

    if offbeam is not None:
        c, _ = np.histogram(offbeam[cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
        scaled_value = c * (exposure['onbeam_events'] / exposure['onbeam_rate']) / (exposure['offbeam_events'] / exposure['offbeam_rate'])
        content += scaled_value

    error = np.sqrt(np.diag(covariance))
    if cfg.get('normalize', False):
        norm = np.sum(content)
        content = content / norm
        error = error / norm

    add_error_boxes(ax, centers, content, width/2.0, error, facecolor='gray', edgecolor='none', alpha=0.5, hatch='///')

    h.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.5, hatch='///'))
    l.append(syslabel)
    ax.legend(h, l, ncol=cfg['legend_ncol'])
    return h, l

def add_data(ax, data, cfg, h, l):
    """
    Adds the data to the plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to plot on.
    data: dict
        The dictionary containing the datasets to plot.
    cfg: dict
        The configuration dictionary for the plot.
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.

    Returns
    -------
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    """
    c, e = np.histogram(data[cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
    err = np.sqrt(c)
    if cfg.get('normalize', False):
        norm = np.sum(c)
        c = c / norm
        err = err / norm
    h.append(ax.errorbar((e[1:] + e[:-1]) / 2.0, c, xerr=np.diff(e) / 2.0, yerr=err, fmt='o', color='black'))
    if cfg.get('normalize', False):
        l.append('Run 2 Data')
    else:
        l.append(f'Run 2 Data ({np.sum(c)})')

    ax.legend(h, l, ncol=cfg['legend_ncol'])
    return h, l

def add_datamc_ratio(ax, data, cfg, exposure, h, l):
    """
    Adds the data/MC ratio to the plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to plot on.
    df: dict
        The dictionary containing the datasets to plot.
    cfg: dict
        The configuration dictionary for the plot.
    exposure: dict
        The dictionary containing the exposure information.
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.

    Returns
    -------
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    """
    systematic, syslabel = cfg['systematic']
    covariance = data[f'systematics_{cfg["channel"]}'][f'{systematic}_{cfg["var"]}']

    c, e = np.histogram(data[f'{cfg["type"]}_{cfg["channel"]}'][cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
    centers = (e[1:] + e[:-1]) / 2.0
    width = np.diff(e)
    scale = (exposure['data'] / exposure['montecarlo'])
    content = scale * c
    data_content, _ = np.histogram(data[f'data_{cfg["channel"]}'][cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])

    error = np.divide(np.sqrt(np.diag(covariance)), content, where=content != 0)
    add_error_boxes(ax, centers, np.repeat(1, len(centers)), width/2.0, error, facecolor='gray', edgecolor='none', alpha=0.5, hatch='///')
    obsratio = np.divide(data_content, content, where=content != 0)
    obsratio[((data_content == 0) & (content == 0))] = 1.0
    mask = (content != 0) & (data_content != 0)
    ratio_error = np.divide(np.sqrt(data_content), content, where=content != 0)
    ax.errorbar(centers[mask], obsratio[mask], xerr=width[mask]/2.0, yerr=ratio_error[mask], fmt='o', color='black')

    ax.set_ylim(0, 2)
    ax.set_ylabel('Obs. / Pred.')

def add_systematic_ratio(ax, data, cfg, exposure, h, l, offbeam=None):
    """
    Adds the systematic variation ratio to the plot.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to plot on.
    df: dict
        The dictionary containing the datasets to plot.
    cfg: dict
        The configuration dictionary for the plot.
    exposure: dict
        The dictionary containing the exposure information.
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    offbeam: pd.DataFrame
        The offbeam data for the plot.

    Returns
    -------
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    """
    systematic, syslabel = cfg['systematic']

    error = np.sqrt(np.diagonal(data[f'systematics_{cfg["channel"]}'][f'{systematic}_{cfg["var"]}']))
    c, e = np.histogram(data[f'{cfg["type"]}_{cfg["channel"]}'][cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
    centers = (e[1:] + e[:-1]) / 2.0
    width = np.diff(e)
    content = (exposure['data'] / exposure['montecarlo']) * c

    if offbeam is not None:
        c, _ = np.histogram(offbeam[cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
        scaled_value = c * (exposure['onbeam_events'] / exposure['onbeam_rate']) / (exposure['offbeam_events'] / exposure['offbeam_rate'])
        content += scaled_value

    ax.errorbar(centers, 100*np.divide(error, content, where=content!=0, ), xerr=width[0] / 2, yerr=0, fmt='_', c='black')
    ax.set_ylim(0, 25)
    ax.set_ylabel('Rel. Error [%]')

    """
    ratio = data[f'systematics_{cfg["channel"]}'][f'{systematic}_{cfg["var"]}_ratio']
    undefined = (ratio == 1)
    error = np.sqrt(np.diagonal(data[f'systematics_{cfg["channel"]}'][f'{systematic}_{cfg["var"]}_cratio']))

    c, e = np.histogram(data[f'{cfg["type"]}_{cfg["channel"]}'][cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])
    centers = (e[1:] + e[:-1]) / 2.0

    ax.errorbar(centers[~undefined], ratio[~undefined], xerr=(np.diff(e) / 2.0)[~undefined], yerr=error[~undefined], fmt='o', color='black')
    ax.axhline(1, color='black', linestyle='--')
    ax.set_ylim(0.8, 1.2)
    ax.set_ylabel('Var. / Nom.')
    """

def calc_chi2(ax, data, cfg, exposure, h, l):
    """
    Calculates the chi2 test statistic between the data and the MC
    using the provided systematic covariance matrix.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis object to plot on.
    data: dict
        The dictionary containing the datasets to plot.
    cfg: dict
        The configuration dictionary for the plot.
    exposure: dict
        The dictionary containing the exposure information.
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.

    Returns
    -------
    h: list[matplotlib.patches.Patch]
        The list of histogram patches.
    l: list[str]
        The list of labels for the histogram patches.
    """
    systematic, syslabel = cfg['systematic']
    covariance = data[f'systematics_{cfg["channel"]}'][f'{systematic}_{cfg["var"]}']
    scale = (exposure['data'] / exposure['montecarlo'])

    content_data = np.histogram(data[f'data_{cfg["channel"]}'][cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])[0]
    content_mc = scale * np.histogram(data[f'{cfg["type"]}_{cfg["channel"]}'][cfg['var']], bins=int(cfg['bins'][0]), range=cfg['bins'][1:])[0]
    total_covariance = covariance + np.diag(content_data)
    mask = content_mc != 0
    chi2 = (content_data[mask] - content_mc[mask]) @ np.linalg.inv(total_covariance[:,mask][mask,:]) @ (content_data[mask] - content_mc[mask]).transpose()
    h.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    l.append(f'$\\chi^2$ / ndof = {chi2:.2f} / {np.sum(mask) - 1}')
    ax.legend(h, l, ncol=cfg['legend_ncol'])
    return h, l

def calculate_full_pureff(data, cfg, write=None):
    """
    Populates a DataFrame with the purity and efficiency of the
    selections at each cut level. The DataFrame is then written as a
    LaTeX table.

    Parameters
    ----------
    data: dict
        The dictionary containing the datasets.
    cfg: dict
        The configuration dictionary for the table.
    write: str
        The path to write the LaTeX table to.

    Returns
    -------
    None.
    """
    results = {'Selection Cut': cfg['cut_labels'],}
    for channel, label in {'1mu1p': '$1\mu1p$', '1muNp': '$1\mu Np$', '1muX': r'$\nu_\mu$ CC'}.items():
        
        cuts = cfg['cut_labels']
        efficiency = TEfficiency('eff', 'Efficiency', len(cuts), 0, len(cuts))
        purity = TEfficiency('pur', 'Purity', len(cuts), 0, len(cuts))

        signal = lambda x, i : int(np.sum(x[i,cfg[channel]['signal_bins']]))
        signal_events = signal(data[f'efficiency_table_{channel}'], 0)
        for ki in range(len(cuts)):
            efficiency.SetTotalEvents(ki+1, signal_events)
            efficiency.SetPassedEvents(ki+1, signal(data[f'efficiency_table_{channel}'], ki))
            purity.SetTotalEvents(ki+1, int(np.sum(data[f'purity_table_{channel}'][ki])))
            purity.SetPassedEvents(ki+1, signal(data[f'purity_table_{channel}'], ki))
        efficiency.SetStatisticOption(6)

        clean = lambda x : 100 if np.abs(x - 100) < 0.01 else x
        results[fr'{label} Purity [\%]'] = [100*purity.GetEfficiency(i) for i in range(1, len(cuts)+1)]
        results[fr'{label} Efficiency [\%]'] = [clean(100*efficiency.GetEfficiency(i)) for i in range(1, len(cuts)+1)]

    df = pd.DataFrame(results)
    latex = (df.style
             .hide(axis="index")
             .format({  r'$1\mu1p$ Purity [\%]': '{:.1f}', r'$1\mu1p$ Efficiency [\%]': '{:.1f}', 
                        r'$1\mu Np$ Purity [\%]': '{:.1f}', r'$1\mu Np$ Efficiency [\%]': '{:.1f}',
                        r'$\nu_\mu$ CC Purity [\%]': '{:.1f}', r'$\nu_\mu$ CC Efficiency [\%]': '{:.1f}'})
             .to_latex(position_float='centering', hrules=True, column_format='ccccccc'))
    lines = latex.splitlines()
    end = lines.index(r'\end{tabular}')
    latex = '\n'.join(lines[2:-1])
    if write is not None:
        with open(write, 'w') as f:
            f.write(latex)

def print_statistics(data, cfg, exposure, write=None):
    """
    Produces a LaTeX table summarizing the systematic uncertainties
    associated with a particular systematic type for each signal
    channel.

    Parameters
    ----------
    data: dict
        The dictionary containing the datasets.
    cfg: dict
        The configuration dictionary for the systematics.
    exposure: dict
        The dictionary containing the exposure information.
    write: str
        The path to write the LaTeX table to.

    Returns
    -------
    None.
    """
    scale_cv = (exposure['data'] / exposure['montecarlo'])

    cmap = {'1mu1p': '$1\mu1p$ [\%]', '1muNp': '$1\mu Np$ [\%]', '1muX': r'$\nu_\mu$ CC [\%]'}
    results = {cfg['columns'][0]: list(), cfg['columns'][1]: list(), '$1\mu1p$ [\%]': list(), '$1\mu Np$ [\%]': list(), r'$\nu_\mu$ CC [\%]': list()}
    bins = [25, 0, 3000]

    for channel in ['1mu1p', '1muNp', '1muX']:
        sim = scale_cv * np.histogram(data[f'selected_{channel}']['reco_visible_energy'], bins=int(bins[0]), range=bins[1:])[0]
        syskeys = [f'{k}_reco_visible_energy' for k in cfg['entries'].keys()]
        names = [cfg['entries'][k][0] for k in cfg['entries'].keys()]
        types = [cfg['entries'][k][1] for k in cfg['entries'].keys()]
        for k in syskeys:
            if channel == '1mu1p':
                results[cfg['columns'][0]].append(names[syskeys.index(k)])
                results[cfg['columns'][1]].append(types[syskeys.index(k)])
            if k not in data[f'systematics_{channel}']:
                results[cmap[channel]].append(0)
                continue
            results[cmap[channel]].append(100*np.sum(np.sqrt(np.diag(data[f'systematics_{channel}'][k])))/np.sum(sim))
    df = pd.DataFrame(results)
    latex = (df.style
             .hide(axis="index")
             .format({'$1\mu1p$ [\%]': '{:.1f}','$1\mu Np$ [\%]': '{:.1f}',r'$\nu_\mu$ CC [\%]': '{:.1f}'})
             .to_latex(position_float='centering', hrules=True, column_format='ccccc'))
    lines = latex.splitlines()
    end = lines.index(r'\bottomrule')
    lines.insert(end-1, r'\midrule')
    end = lines.index(r'\end{tabular}')
    latex = '\n'.join(lines[2:-1])
    if write is not None:
        with open(write, 'w') as f:
            f.write(latex)

def plot_histogram(df, cfg, exposure, dir='reco', save_path=None, release=None):

    if dir is not None:
        cfg['var'] = f'{dir}_{cfg["var"]}'

    if cfg.get('second_axis', None) is not None:
        figure = plt.figure(figsize=(8, 8))
        gspec = figure.add_gridspec(2, 1, height_ratios=[4, 1])
        rax = figure.add_subplot(gspec[1])
        ax = figure.add_subplot(gspec[0], sharex=rax)
    else:
        figure = plt.figure(figsize=(8, 6))
        ax = figure.add_subplot()

    cfg['ylim'] = cfg['ylim'][cfg['channel']]

    if cfg['type'] == 'signal':
        offbeam = None
    else:
        offbeam = df[f'offbeam_{cfg["channel"]}']
    h, l = add_stacked_histogram(ax, df[f'{cfg["type"]}_{cfg["channel"]}'], cfg, exposure, offbeam=offbeam)
    
    ax.set_xlim(*cfg['bins'][1:])
    ax.set_ylim(*(cfg['ylim']))
    ax.set_xlabel(cfg['xlabel'])
    ax.set_ylabel(cfg['ylabel'])
    ax.set_title(cfg['title'])

    if cfg.get('data', False):
        add_data(ax, df[f'data_{cfg["channel"]}'], cfg, h, l)
        h, l = add_systematic(ax, df, cfg, exposure, h, l, offbeam=offbeam)

    #if cfg.get('second_axis', None) in ['datamc_ratio', 'systematics']:
    #    h, l = add_systematic(ax, df, cfg, exposure, h, l)

    if cfg.get('second_axis', None) == 'datamc_ratio' and not cfg.get('normalize', False):
        add_datamc_ratio(rax, df, cfg, exposure, h, l)
        h, l = calc_chi2(ax, df, cfg, exposure, h, l)
        rax.set_xlabel(cfg['xlabel'])
        rax.set_xlim(*cfg['bins'][1:])
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel('')
        #mark_icarus_preliminary(ax, release=None, simulation=False)

    if cfg.get('second_axis', None) == 'pureff':
        add_purity_efficiency(rax, df, cfg)
        rax.set_xlabel(cfg['xlabel'])
        rax.set_xlim(*cfg['bins'][1:])
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel('')
        #mark_icarus_preliminary(ax, release=release, simulation=True)

    if cfg.get('second_axis', None) == 'systematics':
        h, l = add_systematic(ax, df, cfg, exposure, h, l, offbeam=offbeam)
        add_systematic_ratio(rax, df, cfg, exposure, h, l, offbeam=offbeam)
        rax.set_xlabel(cfg['xlabel'])
        rax.set_xlim(*cfg['bins'][1:])
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel('')
        #mark_icarus_preliminary(ax, release=release)

    #if cfg.get('second_axis', None) == None:
        #mark_icarus_preliminary(ax, release=release, simulation=True)

    mark_icarus_preliminary(ax, cfg['watermark'])
    
    if not cfg.get('normalize', False):
        mark_pot(ax, exposure['data'])
    
    if dir is not None:
        cfg['var'] = cfg['var'][5:]

    #if cfg.get('normalize', False):
    #    ax.set_ylim(None)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close(figure)

def make_signal_plots(data, cfg):
    all = {f'{plot_name}_{var_name}': dict(**plot_cfg, **var_cfg) for plot_name, plot_cfg in cfg['plots'].items() for var_name, var_cfg in cfg['variables'].items()}
    keys = [k for k in all.keys() if 'signal_' in k]
    for k in keys:
        dir = 'true' if 'softmax' not in k else None
        plot_histogram(data, all[k], cfg['exposure'], dir=dir, release=None, save_path=f'/Users/mueller/Projects/GitRepos/Thesis/figures/neutrino_selection/{k}.pdf')

def make_selection_plots(data, cfg):
    all = {f'{plot_name}_{var_name}': dict(**plot_cfg, **var_cfg) for plot_name, plot_cfg in cfg['plots'].items() for var_name, var_cfg in cfg['variables'].items()}
    keys = [k for k in all.keys() if 'selected_' in k]
    for k in keys:
        if 'softmax' not in k:
            plot_histogram(data, all[k], cfg['exposure'], dir='reco', release=None, save_path=f'/Users/mueller/Projects/GitRepos/Thesis/figures/neutrino_selection/{k}.pdf')

def make_datamc_plots(data, cfg):
    all = {f'{plot_name}_{var_name}': dict(**plot_cfg, **var_cfg) for plot_name, plot_cfg in cfg['plots'].items() for var_name, var_cfg in cfg['variables'].items()}
    keys = [k for k in all.keys() if 'datamc_' in k]
    for k in keys:
        dir = 'reco' if 'softmax' not in k else None
        plot_histogram(data, all[k], cfg['exposure'], dir=dir, release=None, save_path=f'/Users/mueller/Projects/GitRepos/Thesis/figures/data_mc_comparisons/{k}.pdf')

def make_pureff_table(data, cfg):
    calculate_full_pureff(data, cfg, write='/Users/mueller/Projects/GitRepos/Thesis/figures/neutrino_selection/purity_efficiency.tex')

def make_systematics_table(data, cfg):
    path = '/Users/mueller/Projects/GitRepos/Thesis/figures/systematics/'
    print_statistics(data, cfg['systematics']['detector'], cfg['exposure'], path+'table_detector_systematics.tex')
    print_statistics(data, cfg['systematics']['flux'], cfg['exposure'], path+'table_flux_systematics.tex')
    print_statistics(data, cfg['systematics']['crosssection'], cfg['exposure'], path+'table_crosssection_systematics.tex')
    print_statistics(data, cfg['systematics']['overall'], cfg['exposure'], path+'table_overall_systematics.tex')

def plot_migration(data, cfg, exposure):

    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot()

    ax.hist2d(data['true_'+cfg['var']], data[cfg['var']], bins=cfg['bins'][0], range=[cfg['bins'][1:], cfg['bins'][1:]], cmap='Blues', cmin=1)

def plot_bias(data, cfg):

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()
    bias = np.divide(data[cfg['var']] - data['true_'+cfg['var']], data['true_'+cfg['var']], where=data['true_'+cfg['var']] != 0)
    ax.hist(bias, bins=4*cfg['bins'][0], range=(-0.5,0.5), histtype='step', color='black')