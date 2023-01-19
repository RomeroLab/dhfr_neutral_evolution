import numpy as np
import scipy
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from matplotlib.legend_handler import HandlerPatch
from matplotlib.collections import PatchCollection

import config

WT_PROTEIN = config.WT_AA


def plot_biases_on_axis(ax, b, idx, set_ylabel=True):
    """ b is a vector of biases to plot (length = AA_L)
        idx is an integer which specifies what index this is. 
            It is used to set the title.
    """
    ax.plot(config.AAs, b, "o")
    ax.set_title(f"Local fields for site={idx}")
    if set_ylabel:
        ax.set_ylabel(f"Local field values $h_i(a)$")
    ax.grid()
    return ax

def plot_weights_on_axis(fig, ax, weights_ij, i_idx, j_idx, cmap="bwr", 
        center_cmap=False, aspect=None, axtitle="Coupling values"):
    """ Plot the coupling values on axis """
    
    vmin, vmax = None, None
    if center_cmap:
        # set the scale
        weights_ij_abs_max = np.abs(weights_ij).max()
        vmin=-weights_ij_abs_max*1.2
        vmax=weights_ij_abs_max*1.2
    im = ax.imshow(weights_ij, cmap=cmap, 
            vmax=vmax, vmin=vmin, aspect=aspect)
    ax.set_yticks(range(config.AA_L))
    ax.set_xticks(range(config.AA_L))
    ax.set_xticklabels(config.AAs_string)
    ax.set_yticklabels(config.AAs_string)
    ax.set_ylabel(f"i = {i_idx}")
    ax.set_xlabel(f"j = {j_idx}")

    # Major ticks
    ax.set_xticks(np.arange(0, config.AA_L, 1));
    ax.set_yticks(np.arange(0, config.AA_L, 1));

    # Labels for major ticks
    ax.set_xticklabels(config.AAs_string);
    ax.set_yticklabels(config.AAs_string);

    # Minor ticks
    ax.set_xticks(np.arange(-.5, config.AA_L, 1), minor=True);
    ax.set_yticks(np.arange(-.5, config.AA_L, 1), minor=True);

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.5)

    rect = mpatches.Rectangle(
            (config.AA_MAP[WT_PROTEIN[j_idx]] - 0.5,
             config.AA_MAP[WT_PROTEIN[i_idx]] - 0.5),1,1,
             linewidth=2,edgecolor='black',facecolor='none')
    ax.set_title(axtitle)
    # Add the patch to the Axes
    ax.add_patch(rect)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return (fig, ax)

def plot_distance_array(ax, dist_array, title=""):
    """ Plots a 1-d array of distances (non-negative integers)
        Also overlays a 
    """
    dist_array_max = dist_array.max()
    # Create bins with integer scales so that Matplolib
    # can compute the correct density
    bins = np.arange(0, dist_array_max + 1.5) - 0.5

    ax.hist(dist_array, bins=bins, rwidth=0.2, density=True);
    ax.set_xticks(bins + 0.5)

    # overlay a poisson distribution with lambda = mean
    lam = dist_array.mean()
    pd = scipy.stats.poisson(lam)
    ax.plot(np.arange(dist_array_max), pd.pmf(np.arange(dist_array_max)), '-o')
    ax.set_xlabel("Distance from WT")
    ax.set_ylabel("Probability Density")
    if title:
        ax.set_title(title)
    ax.text(0.75, 0.8, f"mean={dist_array.mean():.2f}\n"
                       f"var ={dist_array.var():.2f}", 
             fontsize=12, transform = ax.transAxes, 
             bbox = dict(boxstyle="round", fc="0.8"));

    
    



# this class copied from 
# https://stackoverflow.com/questions/27826064/matplotlib-make-legend-keys-square/29364054
class HandlerSquare(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = xdescent + 0.5 * (width - height), ydescent
        p = mpatches.Rectangle(xy=center, width=height,
                               height=height, angle=0.0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]      

def plot_top_contacts(top_contacts_idx1, top_contacts_idx2, contact_map, fig=None):
    """ contact_map is a matrix with contacts. 
    (output of get_DHFR_contact_matrix)"""
    # little rectanges for the true and false positives
    matched_rects = []
    unmatched_rects = []
    for i, j in zip(top_contacts_idx1, top_contacts_idx2):
        if contact_map[i, j]:
            rects = matched_rects
        else:
            rects = unmatched_rects
        rects.append(mpatches.Rectangle((i - 0.5, j - 0.5), 1, 1))
        rects.append(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1))


    ax = None
    if fig is None:
        fig, ax = plt.subplots(figsize=(15,15))
    if ax is None:
        ax = fig.gca()

    cmap_labels = ["No contact", r"<5Å", r"5-8Å"]
    cmap = mcolors.ListedColormap(['white', 'turquoise', 'grey'])
    bounds = [0,0.01,5.01,10]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(contact_map, cmap=cmap, norm=norm, origin="upper")

    pc_matched = PatchCollection(matched_rects, edgecolors='red',
            facecolors='none', transOffset=ax.transData)
    pc_unmatched = PatchCollection(unmatched_rects, edgecolors='blue',
            facecolors='none', transOffset=ax.transData)
    ax.add_collection(pc_matched)
    ax.add_collection(pc_unmatched)


    patches = [mpatches.Patch(facecolor=cmap.colors[i+1], label=l,
        edgecolor='lightgrey') for i, l in enumerate(cmap_labels[1:]) ]
    patches.append(mpatches.Patch(label='disagree', edgecolor='blue',
        facecolor='none'))
    patches.append(mpatches.Patch(label='agree', edgecolor='red',
        facecolor='none'))

    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., 
               handler_map={p:HandlerSquare() for p in patches} )

    return(fig)

def save_eps_file(*args, **kwargs):
    """ignore warnings such as
    * The PostScript backend does not support transparency; partially
    transparent artists will be rendered opaque.
    """
    plt.set_loglevel("error")
    plt.savefig(*args, **kwargs)
    plt.set_loglevel("warning")


