import re
import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
from scipy import stats
import warnings
from typing import List
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_mean_intrareplicate(adata: sc.AnnData, 
                                  column: str, 
                                  nonzero: bool = False, 
                                  mean: bool = True, 
                                  quantile: int = None):
       
    #creating an empty dataframe to store the mean values(columns as ions and index as refcol value
    mean_intra = pd.DataFrame(columns=adata.var_names, index=adata.obs[column].cat.categories).astype('float64')
   
    #used loop to calculate mean value of each ion for each replicate
    if nonzero is False:
        for clust in adata.obs[column].cat.categories:
            mean_intra.loc[clust] = adata[adata.obs[column].isin([clust]),:].X.mean(0)
    else:
        if mean is True:
            for clust in adata.obs[column].cat.categories:
                filtered_arr = adata[adata.obs[column].isin([clust]),:].X != 0
            
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_intra.loc[clust] = adata[adata.obs[column].isin([clust]),:].X.mean(0, where=filtered_arr)
                    mean_intra= mean_intra.fillna(0)
        else:
            for clust in adata.obs[column].cat.categories:
                array = adata[adata.obs[column].isin([clust]),:].X.copy()
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    array[array == 0] = np.nan #to use when selecting non-zero value
                    mean_intra.loc[clust] = np.nanpercentile(a=array, q=quantile, axis=0)  
                    mean_intra = mean_intra.astype('float64')

    return mean_intra 


def calculate_correlation(mean_replicate: pd.DataFrame): 
    
    df = mean_replicate.transpose().apply(pd.to_numeric) 
    
    corr_list = []
    columns = df.columns.values #create an array with the column names (ions)
    for index, column in enumerate(columns[:-1]): #using columns as reference reads all values excluding the first one to apply the loop
        slide_cond_rep = re.findall('_.*_', column)[0] #find all values in columns that have the regex for condition (HeLa, NIH..)
        remaining_columns = [s for s in columns[index+1:] if re.match(f'.*{slide_cond_rep}.*', s)] # creates columns_2 based on columns from second value and match only the ones that have the same cell line regex
        for comp_column in remaining_columns: #for each value in remaining_columns
            #calculate pearson correlation and slope
            pearsonR = stats.pearsonr(df[column], df[comp_column], alternative = 'two-sided')
            slope = stats.linregress(x=df[column],y=df[comp_column])
            
            corr_list.append({'col1': column, 'col2': comp_column, 'R': pearsonR.statistic, 'slope':slope[0]})

    corr = pd.DataFrame(corr_list)
    
    #slicing categorical values in col1 to generate condition, slide, and replicate columns
    corr['condition'] = corr.col1.apply(lambda x: re.findall('_.*_', x)[0].replace('_', ''))
    corr['slide_col1'] = corr.col1.apply(lambda x: re.findall('^[A-Za-z0-9]+', x)[0])
    corr['slide_col2'] = corr.col2.apply(lambda x: re.findall('^[A-Za-z0-9]+', x)[0])
    corr['rep_col1'] = corr.col1.apply(lambda x: re.findall('[0-9]+$', x)[0])

    return corr
    

def calculate_CV(mean_replicate: pd.DataFrame, key: List[str], column: str):
    index_name = mean_replicate.index.name
    mean_replicate.columns.name = None
    if index_name == None:
        index_name = 'index'
    df = mean_replicate.reset_index().melt(id_vars=index_name,
                                           value_vars=mean_replicate.columns[1:]).rename(columns={'variable':'ion',
                                                                                                  index_name:column})
    df['condition'] = df[column].apply(lambda x: re.findall('_.*_', x)[0].replace('_', ''))
    df['slide'] = df[column].apply(lambda x: re.findall('^[A-Za-z0-9]+', x)[0])
    df_mean = df.groupby(key)['value'].mean().to_frame()
    df_SD = df.groupby(key)['value'].std().to_frame()
    df_CV = df_SD.div(df_mean).multiply(100)
    df_metrics = df_mean.merge(df_CV,
                               on = key).reset_index().rename(columns={'value_x':'mean',
                                                                       'value_y':'CV'})
    return df_metrics
    

def compile_differential_analysis(adata: sc.AnnData, column: str, reference: str) -> pd.DataFrame:

    conds_to_plot = adata.obs[column].unique()[adata.obs[column].unique() != reference]
    ion = []
    for idx, i in enumerate(conds_to_plot):
        score_cell = i+'_scores'
        scores = pd.DataFrame(adata.uns['rank_genes_groups']['scores'][i])
        scores = scores.set_index(adata.uns['rank_genes_groups']['names'][i])
        scores = scores.rename(columns={0: score_cell})
    
        log2FC_cell = i+'_log2FC'
        log2FC = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'][i])
        log2FC = log2FC.set_index(adata.uns['rank_genes_groups']['names'][i])
        log2FC = log2FC.rename(columns={0: log2FC_cell})
    
        pvals_cell = i+'_pvals_adj'
        pvals_adj = pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj'][i])
        pvals_adj = pvals_adj.set_index(adata.uns['rank_genes_groups']['names'][i])
        pvals_adj = pvals_adj.rename(columns={0: pvals_cell})
    
        pts_cell = i+'_pts'
        pts = pd.DataFrame(adata.uns['rank_genes_groups']['pts'][i])
        pts = pts.set_index(adata.uns['rank_genes_groups']['names'][i])
        pts = pts.rename(columns={i: pts_cell})
    
        ptsrest_cell = i+'_pts_rest'
        ptsrest = pd.DataFrame(adata.uns['rank_genes_groups']['pts_rest'][i])
        ptsrest = ptsrest.set_index(adata.uns['rank_genes_groups']['names'][i])
        ptsrest = ptsrest.rename(columns={i: ptsrest_cell})
    
        DA = pd.concat(objs=[scores,log2FC,pvals_adj,pts,ptsrest], axis = 1)
        DA = DA.reset_index().rename(columns={'index': 'ion'})
        ion.extend(DA.to_dict('records'))

    DA_dict = {}
    for item in ion:
        key = item['ion']
        item.pop(key, None)
        if key in DA_dict:
            DA_dict[key].update(item)
        else:
            DA_dict[key] = item
        
    DA_df = pd.DataFrame.from_dict(DA_dict, orient='index')

    return DA_df


def checking_isomers(bulk: pd.DataFrame, SC: pd.DataFrame):
    #checking the presence of isomers (2 columns with same formula) 
    cols = pd.Series(bulk.columns) #listing column names of bulk data
    #renaming duplicates present in the bulk dataset with .1, .2 (per order of appearance)
    for dup in bulk.columns[bulk.columns.duplicated(keep = False)]:
        cols[bulk.columns.get_loc(dup)] = ([dup + '.' + str(d_idx)
                                            if d_idx !=0
                                            else dup
                                            for d_idx in range(bulk.columns.get_loc(dup).sum())])
    bulk.columns = cols #getting a list of bulk columns
    bulk.head()

    #duplicating columns in sc dataset based on the columns of bulk to be able to merge datasets
    for column in cols:
        if re.match(r'.*\.[0-9]*$', column):
            orig_column = re.findall('.+?(?=\.[0-9])', column)[0]
            if orig_column in SC.columns:
                SC[column] = SC[orig_column]
    return bulk, SC

    
def prepare_ORA_file(adata: sc.AnnData, column: str, reference: str):
    
    ORA_path =  Path(r'../data') / 'ORA'
    ORA_path.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = ORA_path
    
    conds_to_plot = adata.obs[column].unique()[adata.obs[column].unique() != reference]
    ion = []
    for idx, i in enumerate(conds_to_plot):    
        log2FC_cell = 'log2FC'
        log2FC = pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges'][i])
        log2FC = log2FC.set_index(adata.uns['rank_genes_groups']['names'][i])
        log2FC = log2FC.rename(columns={0: log2FC_cell})
    
        pvals_cell = 'pvals_adj'
        pvals_adj = pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj'][i])
        pvals_adj = pvals_adj.set_index(adata.uns['rank_genes_groups']['names'][i])
        pvals_adj = pvals_adj.rename(columns={0: pvals_cell})

        DA = pd.concat(objs=[log2FC,pvals_adj], axis = 1)
    
        pos_ions = DA.loc[(DA['pvals_adj']<= 0.05)] 
        pos_ions = pos_ions.loc[(pos_ions['log2FC']>= 1.5)] 
        pos_ions = np.array(pos_ions.index).astype(str) #get only names without scores, as np.array
        pos_ions = np.char.rstrip(pos_ions, '-H') #remove adduct from name
    
        neg_ions = DA.loc[(DA['pvals_adj']<= 0.05)] 
        neg_ions = neg_ions.loc[(neg_ions['log2FC']<= -1.5)]
        neg_ions = np.array(neg_ions.index).astype(str) #get only names without scores, as np.array
        neg_ions = np.char.rstrip(neg_ions, '-H') #remove adduct from name
    
        #store top formulas as csv's for ORA (=over representation analysis) to use in RStudio (bmetenrichr package)
        np.savetxt(fname = ORA_path / 'SC2_{}_formulas_top-up.csv'.format(i), 
                   X = pos_ions,
                   delimiter=",",
                   fmt="'%s',",
                   newline="")
        np.savetxt(fname = ORA_path/ 'SC2_{}_formulas_top-down.csv'.format(i), 
                   X = neg_ions,
                   delimiter=",",
                   fmt="'%s',",
                   newline="")

    all_formulas = np.array(DA.index).astype(str) #get only names without scores, as np.array
    all_formulas = np.char.rstrip(all_formulas, "-H") #remove adduct from name
    np.savetxt(fname = ORA_path / 'SC2_formula_background.csv'.format(i),
                   X = all_formulas,
                   delimiter=",",
                   fmt="'%s',",
                   newline="")
    return

def calculate_cell_dropout(adata: sc.AnnData, cell: str) -> pd.DataFrame:
    int_matrix = pd.DataFrame(columns=adata.var_names, data = adata.X, index = adata.obs_names)
    int_matrix[int_matrix != 0] = 1
    int_matrix = int_matrix.astype(int)

    result = {}
    columns = []
    for column1 in int_matrix.columns:
        for column2 in int_matrix.columns:
            if column1 != column2:
                new_column = ','.join(sorted([column1, column2]))
                result[new_column] = int_matrix[column1] + int_matrix[column2]
                columns.append(new_column)
    result_df = pd.concat(result.values(), axis=1, ignore_index=True)
    result_df.columns = result.keys()
    result_df['cell_line'] = cell
    result_df = result_df.melt(id_vars=['cell_line'], value_vars=columns, var_name='ions')
    result_df['cell_count'] = 0    
    result_df['value'] = pd.Categorical(result_df['value'])
    result_df = result_df.groupby(['cell_line', 'ions', 'value'], observed=False).count().reset_index()
    result_df['proportion'] = result_df['cell_count'] / len(int_matrix)
    return result_df

def ion_correlation_calculation(int_matrix: pd.DataFrame, cell: str) -> pd.DataFrame:
    corr = []
    columns = int_matrix.columns.values
    for idx_1, col_1 in enumerate(columns[:-1]):
        for col_2 in columns[idx_1 + 1:]:
            res = stats.pearsonr(int_matrix[col_1], int_matrix[col_2], alternative='two-sided')
            new_column = ','.join(sorted([col_1, col_2]))
            corr.append({'cell_line': cell, 'ions': new_column, 'col1': col_1, 'col2': col_2, 'corr': res.statistic, 'p-value':res.pvalue})

    return corr

def calculate_metrics(data: pd.DataFrame, sets: dict, cell_line: str, column1='col1', column2='col2'):
    members = list(data["col1"].unique())
    members.extend(list(data["col2"].unique()))
    members_tot = len(list(dict.fromkeys(members)))

    lst_metrics = []
    for index, value in sets.items():
        filter1 = data[column1].isin(value)
        filter2 = data[column2].isin(value)
        comm_df = data[filter1 & filter2]
        
        # Create graph object of this community and calculate metrics
        graph = nx.from_pandas_edgelist(comm_df, column1, column2, edge_attr=None, create_using=nx.Graph())
        set_num = len(value)
        ratio = set_num / members_tot
        lst_metrics.append({
            'cell_line': cell_line,
            'avg_connectivity':nx.average_node_connectivity(graph, flow_func=None),
            'class': index,
            'ions_num': set_num,
            'ions': value,
            })

    return lst_metrics

def get_set_by_class(classname):
    match classname:
        case 'Amino acids, peptides, and analogs':
            return 'Amino acids, peptides, and analogs'
        case 'Lipids and lipid-like molecules':
            return 'Lipids and lipid-like molecules'
        case 'Carbohydrates and carbohydrate conjugates':
            return 'Carbohydrates and carbohydrate conjugates'
        case 'Nucleosides, nucleotides, and analogs':
            return 'Nucleosides, nucleotides, and analogs'
        case 'Hydroxy acids, keto acids, carboxylic acids, and derivatives':
            return 'Hydroxy acids, keto acids, carboxylic acids, and derivatives'
        case 'Purines, pyrimidines, and derivatives':
            return 'Purines, pyrimidines, and derivatives'
        case 'Inorganic phosphates, organic phosphates, and derivatives':
            return 'Inorganic phosphates, organic phosphates, and derivatives'
        case 'Vitamins and cofactors':
            return 'Vitamins and cofactors'
        case 'Others':
            return 'Others'

def get_color_by_class(classname):
    match classname:
        case 'Amino acids, peptides, and analogs':
            return '#332288ff'
        case 'Lipids and lipid-like molecules':
            return '#88cceeff'
        case 'Carbohydrates and carbohydrate conjugates':
            return '#44aa99ff'
        case 'Nucleosides, nucleotides, and analogs':
            return '#117733ff'
        case 'Hydroxy acids, keto acids, carboxylic acids, and derivatives':
            return '#999933ff'
        case 'Purines, pyrimidines, and derivatives':
            return '#ddcc77ff'
        case 'Inorganic phosphates, organic phosphates, and derivatives':
            return '#cc6677ff'
        case 'Vitamins and cofactors':
            return '#882255ff'
        case 'Others':
            return '#aaaaaaff'
        
def insert_set(data, class_set, ion):
    if class_set not in data:
        data[class_set] = frozenset({ion})
    else:
        data[class_set] = data[class_set].union(frozenset({ion}))
    return

def fluo_processing(adata, fluo_column, resolution, plot_figure, bins=None, color=None, limit=None):
    fluo = adata.obs[fluo_column].values #get fluorescence values for all cells 
    gaussian = GaussianMixture(n_components=2).fit(fluo.reshape(-1, 1))
    # Evaluate fluorescence distribution
    x = np.linspace(fluo.min(), fluo.max(), resolution)
    y = np.exp(gaussian.score_samples(x.reshape(-1, 1)))
    # Find local valley
    minima = np.where((y[:-2] > y[1:-1]) & (y[1:-1] < y[2:]))[0] + 1
    threshold = x[minima[-1]]
    if 'condition' not in adata.obs.columns:
        adata.obs['condition'] = 'NCI-H460'
        # Assign 'NIH3T3' to cells where the GFP intensity is > threshold
        adata.obs.loc[adata.obs[fluo_column] > threshold, 'condition'] = 'NIH3T3'
        counts = adata.obs['condition'].value_counts()
        NIH3T3_count = counts.get('NIH3T3', 0)
        NCIH460_count = counts.get('NCI-H460', 0)
        print(f'NIH3T3: {NIH3T3_count}, NCI-H460: {NCIH460_count}')
    else:
        counts = adata.obs['condition'].value_counts()
        NIH3T3_count = counts.get('NIH3T3', 0)
        NCIH460_count = counts.get('NCI-H460', 0)
        print(f'NIH3T3: {NIH3T3_count}, NCI-H460: {NCIH460_count}')
    
    if plot_figure is True:
        # Plot histograms and gaussian curves
        hist_counts, bins = np.histogram(fluo, bins=bins)
        scaling_factor = hist_counts.max() / y.max()
        y_scaled = y * scaling_factor
        # Define the y-axis break point
        upper_limit = hist_counts.max()
        lower_limit = np.percentile(hist_counts, limit)  # adjust if needed
        
        # Create subplots: two rows, shared x-axis
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(5, 4), gridspec_kw={'height_ratios': [0.5, 1.0]})
        
        # Top plot: focus on high range
        sns.histplot(fluo, bins=bins, edgecolor='black', color=color, ax=ax_top, kde=False)
        ax_top.plot(x, y_scaled, color="black", lw=1)
        ax_top.axvline(x=threshold, color='red', linestyle='--', lw=1, label='threshold')
        ax_top.set_ylim(lower_limit, upper_limit * 1.1)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.tick_params(labelbottom=False, labelsize=14)
        ax_top.legend()  # or ax_bottom.legend(), depending on where you want it

        # Bottom plot: focus on low range
        sns.histplot(fluo, bins=bins, edgecolor='black', color=color, ax=ax_bottom, kde=False)
        ax_bottom.plot(x, y_scaled, color="black", lw=1)
        ax_bottom.axvline(x=threshold, color='red', linestyle='--', lw=1)
        ax_bottom.set_ylim(0, lower_limit)
        ax_bottom.spines['top'].set_visible(False)
        ax_bottom.tick_params(labelsize=14)
        
        # Diagonal lines to show the break
        d = .015  # size of diagonal lines
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1.5)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top left diagonal
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top right diagonal
        
        kwargs.update(transform=ax_bottom.transAxes)  # switch to bottom axes
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom left diagonal
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom right diagonal
        
        # Labels and layout
        ax_bottom.set_xlabel('GFP Intensity', fontsize=16)
        ax_top.set_ylabel('')
        ax_bottom.set_ylabel('Cell Count', fontsize=16)
        plt.tight_layout()
        plt.show()
        return fig