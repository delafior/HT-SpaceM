from typing import Dict, Union
from warnings import warn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import to_rgb
from sklearn.mixture import GaussianMixture

def highlight_scatterplot(
    data: pd.DataFrame,
    x=None,
    y=None,
    col: str = None,
    row: str = None,
    obsm_key=None,
    background_color="#cccccc",
    background_points_to_plot: str = "others",
    trim_axes=False,
    decorate_titles: bool = False,
    titles_palette=None,
    scatter_kwargs: Dict = None,
    **kwargs,
) -> sns.FacetGrid:
    """Create a Seaborn ``FacetGrid`` of scatterplots, where in each facet the datapoints not meeting the facet's condition are drawn in a background color rather than being excluded.
    The purpose of such a plot is to make groups of datapoints easier to distiguish while preserving visual information about the group's relative position to other datapoints.
    Out of the box this function behaves like Seaborn's ``FacetGrid``, and most of its parameters (such as ``hue``) are directly accessible by passing them as additional keyword arguments.
    Args:
        data (:class:`pandas.DataFrame` or :class:`anndata.AnnData`):
            Tidy (“long-form”) dataframe where each column is a variable and each row is an observation or an AnnData object.
            When passing an AnnData object, the AnnData's ``.obs`` DataFrame will be used.
        x (:class:`str` or :class:`int`): Variable in ``data.columns`` that specifies positions on the x axis.
            When ``obsm_key`` is passed, ``x`` is instead used as an index to select a column from the respective ``.obsm`` entry (default 1).
            **Note that for consistency with common plotting practices these indices are 1-based!**
        y (:class:`str` or :class:`int`):
            Variable in ``data.columns`` that specifies positions on the y axis.
            When ``obsm_key`` is passed, ``y`` is instead used as an index to select a column from the respective ``.obsm`` entry (default 2).
            **Note that for consistency with common plotting practices these indices are 1-based!**
        row (:class:`str`):
            Variable in ``data.columns`` that defines subsets of the data, which will be drawn on separate facets in the grid.
        col (:class:`str`):
            Variable in ``data.columns`` that defines subsets of the data, which will be drawn on separate facets in the grid.
        obsm_key (:class:`str`, optional):
            When passing an AnnData object as ``data``, use values from ``data.obsm[<obsm_key>]`` instead for the x and y positions of points.
        background_color: (:class:`str` or Matplotlib-compatible color format):
            Color to use for background datapoints.
        background_points_to_plot: (:class:`str`):
            Which datapoints to plot as background datapoints. Allowed values are:
                - 'others' (default): datapoints in all other facets are plotted as background
                - 'none': no background datapoints are plotted
                - 'others_in_same_col', 'others_in_same_row':
                    Datapoints in other facets of the same row/col are plotted as background.
                    In this case, facetgrid kwargs 'sharex' and 'sharey` are set to false by default
                    (but can be overwritten by passing custom values in 'kwargs' argument.)
        trim_axes: (:class:`bool` or :class:`float`):
            Draw minimalist axes: no axis ticks will be draw, x- and y-axis will only be drawn on the bottom-left facet and will be truncated.
            If ``trim_axes`` is a float, will truncate the axes to ``trim_axes`` * a single facet's display width, otherwise to 35% of the width.
        decorate_titles (:class:`bool`):
            Whether to draw colored boxes around the facet's titles.
            Note this is only supported for facetting along either rows or columns, but not both!
        titles_palette (:class:`str`, :class:`dict`, or sequence, optional):
            The color palette to use for the boxes around the titles. Only required when ``decorate_titles = True`` and ``data`` is a DataFrame.
            When data is an AnnData and no color palette is provided, any color palette stored at ``adata.uns["<key_split>_colors"]`` will be used if present,
            otherwise will use the default palette.
        scatter_kwargs (:class:`dict`):
            keyword arguments that will be passed to on to ``seaborn.scatterplot`` for both foreground and background datapoints.
        kwargs:
            Keyword arguments to be passed on to ``seaborn.FacetGrid``.
    Returns:
        :class:`seaborn.FacetGrid`:
            A FacetGrid of scatterplots.
    Examples:
        Using a DataFrame:
        >>> outer_spacem.pl.highlight_scatterplot(
        ...     data=iris,
        ...     x='sepal length (cm)',
        ...     y='sepal width (cm)',
        ...     hue="class",
        ...     col="class",
        ... )
        .. image:: examples/plot_examples/pl.highlight_scatterplot.png
        |
        Using an AnnData, plotting components 1 and 2 of a UMAP:
        >>> outer_spacem.pl.highlight_scatterplot(
        ...     data=adata,
        ...     obsm_key="X_umap", # created using scanpy.tl.umap()
        ...     x=1,
        ...     y=2,
        ...     col="treatment",
        ...     hue="treatment",
        ...     palette=pal,
        ...     decorate_titles=True,
        ...     titles_palette=pal,
        ...     trim_axes=True
        ... )
        .. image:: examples/plot_examples/pl.highlight_scatterplot_b.png
        |
    """
    # Validate 'background_points_to_plot' values:
    assert background_points_to_plot in [
        "others",
        "none",
        "others_in_same_col",
        "others_in_same_row",
    ], (
        f"The value of 'background_points_to_plot'='{background_points_to_plot}' is not valid. "
        f"Allowed values are: 'others', 'none', 'others_in_same_col', 'others_in_same_row'."
    )
    if background_points_to_plot == "others_in_same_col":
        assert col is not None, (
            "A `col` key should be passed when `background_points_to_plot` is set "
            "to 'others_in_same_col'"
        )
    elif background_points_to_plot == "others_in_same_row":
        assert row is not None, (
            "A `row` key should be passed when `background_points_to_plot` is set "
            "to 'others_in_same_row'"
        )
    # Possibly set default sharex/sharey facet-grid values:
    if (
        background_points_to_plot == "others_in_same_col"
        or background_points_to_plot == "others_in_same_row"
    ):
        kwargs.setdefault("sharex", False)
        kwargs.setdefault("sharey", False)
    if scatter_kwargs is None:
        scatter_kwargs = {}
    # Dynamically adapt point size (if none is set)
    scatter_kwargs.setdefault("s", min(25000 / len(data), 50))
    scatter_kwargs.setdefault("linewidth", 0)
    if isinstance(data, sc.AnnData):
        adata = data
        if obsm_key:
            data = adata.obs.copy()
            obsm_name = obsm_key.lstrip("X_").upper()
            if x is None:
                x = 1
            if y is None:
                y = 2
            x_colname = f"{obsm_name}{x}"
            y_colname = f"{obsm_name}{y}"
            data[[x_colname, y_colname]] = adata.obsm[obsm_key][
                :, [x - 1, y - 1]
            ]  # For consistency with PCA/UMAP practices components
            x = x_colname
            y = y_colname
        else:
            data = adata.obs.copy()
        if "hue" in kwargs:
            kwargs.setdefault("palette", _get_palette_dict(adata, kwargs["hue"]))
        if decorate_titles and not isinstance(titles_palette, dict):
            titles_palette = _get_palette_dict(adata, col or row, titles_palette)
    else:
        if obsm_key:
            warn("An obs_key was passed but data is not an AnnData object, ignoring obs_key")
        if x is None or y is None:
            raise TypeError(
                f"Need to provide x and y when passing a DataFrame to highlight_scatterplot()"
            )
    g = sns.FacetGrid(data, col=col, row=row, **kwargs)
    # Draw background points
    if background_points_to_plot != "none":
        if data.index.duplicated().any():
            if background_points_to_plot != "others":
                warn("Data has duplicate index: all datapoints are drawn as background points.")
            # Fallback for when there's a duplicate index: draw all background datapoints
            for ax in g.axes.flatten():
                sns.scatterplot(
                    ax=ax, data=data, x=x, y=y, color=background_color, zorder=0, **scatter_kwargs
                )
        else:
            # Only draw background datapoints
            # facet_data will return a separate axis for each hue, so we have to combine indices
            facet_indices = pd.DataFrame(
                [
                    (row_idx, col_idx, hue, set(data.index))
                    for (row_idx, col_idx, hue), data in g.facet_data()
                ],
                columns=["row", "col", "hue", "index"],
            )
            facet_indices = (
                facet_indices.groupby(["row", "col"])["index"]
                .apply(lambda x: set.union(*x))
                .reset_index()
            )
            for _, (row_idx, col_idx, facet_index) in facet_indices.iterrows():
                ax = g.facet_axis(row_idx, col_idx)
                filtered_data = data.copy()
                if background_points_to_plot == "others_in_same_row":
                    filtered_data = filtered_data[filtered_data[row] == g.row_names[row_idx]]
                    if not ax.get_subplotspec().is_last_col():
                        ax.set_yticklabels([])
                elif background_points_to_plot == "others_in_same_col":
                    filtered_data = filtered_data[filtered_data[col] == g.col_names[col_idx]]
                    if not ax.get_subplotspec().is_last_row():
                        ax.set_xticklabels([])
                background_index = filtered_data.index.difference(facet_index)
                background_data = data.loc[background_index, :]
                sns.scatterplot(
                    ax=ax,
                    data=background_data,
                    x=x,
                    y=y,
                    color=background_color,
                    zorder=0,
                    **scatter_kwargs,
                )
    # Draw highlighted points on top of background
    g.map(sns.scatterplot, x, y, zorder=1, **scatter_kwargs)
    # Minimalist axes
    if not (trim_axes is False):
        extent = 0.35 if trim_axes == True else trim_axes
        corner_ax = list(g._left_axes)[-1]
        for ax in g.axes.flatten():
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove axis labels
            if ax is not corner_ax:
                sns.despine(bottom=True, left=True, ax=ax)
                ax.xaxis.get_label().set_visible(False)
                ax.yaxis.get_label().set_visible(False)
        # Resize axes on corner ax
        xlims = corner_ax.get_xlim()
        ylims = corner_ax.get_ylim()
        # Make corner axis equal display length
        figW, figH = g.fig.get_size_inches()
        _, _, w, h = corner_ax.get_position().bounds  # Axis size on figure
        disp_ratio = (figH * h) / (figW * w)  # Ratio of display units
        data_ratio = np.ptp(ylims) / np.ptp(xlims)  # Ratio of display units to set lims
        aspect = disp_ratio / data_ratio
        corner_ax.spines["bottom"].set_bounds(xlims[0], np.ptp(xlims) * extent + xlims[0])
        corner_ax.spines["left"].set_bounds(ylims[0], np.ptp(ylims) * extent / aspect + ylims[0])
        corner_ax.set_xlabel(corner_ax.get_xlabel(), loc="left")
        corner_ax.set_ylabel(corner_ax.get_ylabel(), loc="bottom")
    if decorate_titles:
        if titles_palette is None:
            raise ValueError(
                "When decorating titles for a DataFrame you need to pass a palette as well"
            )
        _decorate_facet_titles(g, titles_palette)
    return g

def _get_palette_dict(adata, key, pal=None):
    """
    Create a dict from a ``adata.obs`` column mapping unique column values to a color palette.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object from whose `.obs` values shall be mapped.
        key (:class:`str`):
            Column in ``adata.obs`` to map colors to.
        pal (:class:`str`, ``None``, or sequence, optional):
            The color palette to map column values to.
    Returns:
        :class:`dict`:
            A dict mapping unique column entries to a color palette.
            If ``pal==None`` and a ``<key>_colors`` entry exists in ``adata.uns``, will produce a dict mapping column values to the colors stored there instead.
    """

    # NOTE: Might be redundant with sc.pl._tools.scatterplots._get_palette()!

    cat = adata.obs[key]
    cat = cat.cat.categories if cat.dtype.name == "category" else cat.unique()

    if pal is None and f"{key}_colors" in adata.uns:
        pal = adata.uns[f"{key}_colors"]
    else:
        pal = sns.color_palette(pal, n_colors=len(cat))

    pal = dict(zip(cat, pal))
    return pal

def scatterplot_correlation(data, X, Y, color, column=None, observation=None, X_name=None, Y_name=None, filename=None, title=None):
    corr_vals = []
    pvals = []

    if observation is None:
        subset = data
    else:
        subset = data[data[column] == observation]    
    
    for _, group in subset.groupby("well"):
        if group[[X, Y]].dropna().shape[0] >= 2:  # Enough data points
            try:
                res = stats.pearsonr(group[X], group[Y])
                corr_vals.append(res.statistic)
                pvals.append(res.pvalue)
            except Exception:
                continue
    
    # Mean of well-level correlations
    if corr_vals:
        mean_r = np.mean(corr_vals)
        mean_p = np.mean(pvals)
    else:
        mean_r = float("nan")
        mean_p = float("nan")
    
    # Add regression line (over all data in the condition)
    plt.figure(figsize=(4, 4))
    ax=sns.regplot(data=subset,
                x=X,
                y=Y,
                #ax=ax,
                scatter=True,
                color=color,
                ci=95,
                line_kws={'lw': 2, 'color': 'black'},
                scatter_kws={'s': 40,
                             'edgecolor': 'white',
                             'alpha': 1.0},
                truncate=False)
    
    ax.grid(False)
    ax.minorticks_off()
    ax.text(0.05, 0.95,
                f'R: {mean_r:.3f}',
                transform=ax.transAxes,
                fontsize=22,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel(xlabel=X_name, fontsize=24)
    plt.ylabel(ylabel=Y_name, fontsize=24)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    if title:
        ax.set_title(title, fontsize=24)
    if filename:
        plt.savefig(plots_path/f'{filename}_{observation}_{X}_{Y}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(plots_path/f'{filename}_{observation}_{X}_{Y}.png', dpi=300, bbox_inches='tight')
    plt.show()

def histogram(adata, column, resolution, bins, color, limit, size_x, size_y, x_label, threshold):
    fluo = adata.obs[column].values #get fluorescence values for all cells 
    gaussian = GaussianMixture(n_components=2).fit(fluo.reshape(-1, 1))
    # Evaluate fluorescence distribution
    x = np.linspace(fluo.min(), fluo.max(), resolution)
    y = np.exp(gaussian.score_samples(x.reshape(-1, 1)))
    
    # Plot histograms and gaussian curves
    hist_counts, bins = np.histogram(fluo, bins=bins)
    scaling_factor = hist_counts.max() / y.max()
    y_scaled = y * scaling_factor
    # Define the y-axis break point
    upper_limit = hist_counts.max()
    lower_limit = np.percentile(hist_counts, limit)  # adjust if needed
    
    # Create subplots: two rows, shared x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(size_x, size_y), gridspec_kw={'height_ratios': [0.5, 1.0]})
    
    # Top plot: focus on high range
    sns.histplot(fluo, bins=bins, edgecolor='black', color=color, ax=ax_top, kde=False)
    ax_top.plot(x, y_scaled, color="black", lw=1)
    ax_top.axvline(x=threshold, color='red', linestyle='--', lw=1, label='threshold')
    ax_top.set_ylim(lower_limit, upper_limit * 1.1)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(labelbottom=False, labelsize=18)
    ax_top.legend()  # or ax_bottom.legend(), depending on where you want it

    # Bottom plot: focus on low range
    sns.histplot(fluo, bins=bins, edgecolor='black', color=color, ax=ax_bottom, kde=False)
    ax_bottom.plot(x, y_scaled, color="black", lw=1)
    ax_bottom.axvline(x=threshold, color='red', linestyle='--', lw=1)
    ax_bottom.set_ylim(0, lower_limit)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.tick_params(labelsize=18)
    
    # Diagonal lines to show the break
    d = .015  # size of diagonal lines
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1.5)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top right diagonal
    
    kwargs.update(transform=ax_bottom.transAxes)  # switch to bottom axes
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom left diagonal
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom right diagonal
    
    # Labels and layout
    ax_bottom.set_xlabel(f'{x_label}', fontsize=22)
    ax_top.set_ylabel('')
    ax_bottom.set_ylabel('Cell Count', fontsize=22)
    plt.tight_layout()
    plt.show()
    return fig
