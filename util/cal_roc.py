import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pickle
import pandas as pd
import copy
import pdb

# Calculate ROC AUC PPV 
def calConfIntrvl_auc(Y_score,Y_true,n_bootstraps):
    bootstrapped_scores = []
    rng = np.random.RandomState(1)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(Y_score)-1, len(Y_score))
        if len(np.unique(Y_true[indices])) < 2:
            continue
        score = roc_auc_score(Y_true[indices], Y_score[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower, confidence_upper
    
def get_ROC_precisionRecall(Y_true,Y_score,savePath=None,CI_flag=False):
    Y_true = np.array(Y_true)
    if CI_flag:
        auc_CI_lower, auc_CI_upper = calConfIntrvl_auc(Y_score,Y_true,500)
    else:
        auc_CI_lower = 0
        auc_CI_upper = 0
    roc_auc = roc_auc_score(Y_true, Y_score)
    fpr, tpr, thresholds = metrics.roc_curve(Y_true, Y_score)     
    Pr_positive = sum(Y_true)/Y_score.shape[0]
    Pr_negative = 1-Pr_positive
    ppv = Pr_positive*np.array(tpr)/(Pr_positive*np.array(tpr) + Pr_negative*np.array(fpr))
    npv = Pr_negative*np.array(1-fpr)/(Pr_positive*np.array(1-tpr) + Pr_negative*np.array(1-fpr))
    precision_recall_auc = trapz(x=tpr[~np.isnan(np.array(ppv))], y=ppv[~np.isnan(np.array(ppv))])
    Pr_positive = sum(Y_true)/Y_true.shape[0]
    Pr_negative = 1-Pr_positive
    ppr = Pr_positive*tpr+Pr_negative*fpr
    if savePath:
        savePath = '%s_C_%.3g.txt' %(savePath[:-4],roc_auc)
        with open(savePath, 'wb') as fp:
            pickle.dump([roc_auc, fpr, tpr, thresholds, ppv, npv, precision_recall_auc, auc_CI_lower, auc_CI_upper,Y_score,Y_true,Pr_positive,Pr_negative, ppr], fp)    
    return roc_auc, fpr, tpr, thresholds, ppv, npv, precision_recall_auc, auc_CI_lower, auc_CI_upper,Y_score,Y_true,Pr_positive,Pr_negative, ppr

def plotFeatureImportance(df,labelColName,scoreColName,numVars,fullNameSave,xlabel,neg=False):
    if neg:
        df['valTemp'] = np.abs(df[scoreColName])
        df.sort_values(by=['valTemp'],ascending=False,inplace=True)
    else:
        df.sort_values(by=[scoreColName],ascending=False,inplace=True)
    varList = df[labelColName].tolist()
    varScore = np.array(df[scoreColName])
    varList_1_25 = varList[:numVars]
    varScore_1_25 = varScore[:numVars]    
    y_pos = np.arange(len(varList_1_25))
    plt.figure(figsize=(9 ,9), dpi= 100, facecolor='w', edgecolor='k')
    plt.barh(y_pos, varScore_1_25, align='center', color=(0.6,0.6,0.6,0.6), ecolor='black')
    if neg:
        plt.xlim([np.min(varScore_1_25), np.max(varScore_1_25)])
    else:
        plt.xlim([0, np.max(varScore_1_25)])
    plt.ylim([-1, len(varList_1_25)])
    ax = plt.gca()
    plt.gca().xaxis.grid(True)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(varList_1_25, fontsize=12)
    ax.yaxis.labelpad = 100
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
    plt.savefig(fullNameSave, dpi=600,bbox_inches='tight')


def get_idx(arr, x, arr2):
    return (arr[np.argmin(np.abs(arr-x))], arr2[np.argmin(np.abs(arr-x))])
    
def plot_perf_curves(Y_validation, Y_validation_score, savePath, CI_flag=False, 
                     plot_type=None, plot_spec_custom={}, group_col=None, 
                     group_color={}, group_white_list=[0], label_dict_custom={}, 
                     add_lines=[], title=None):
        
    label_dict = {'fpr': '1-Specificity (%)', 'tpr': 'Sensitivity (%)', \
                  'ppv': 'Precision (%)', 'ppv-1': 'Number needed to evaluate', \
                  'ppr': 'Number predicted as positive per 100 patients', \
                  'thresholds': 'prediction score'}

    plot_spec = {"xticks": np.arange(0,101,10),
                 "yticks": np.arange(0,101,10),
                 "xlim": [0, 100],
                 "ylim": [0, 100],
                 "fontsize": 20}

    scale_list = ['fpr', 'tpr', 'fnr', 'tnr', 'ppv', 'npv', 'thresholds']        
        
    print("..................Label & Plot & Scale Initialization..............")
    
    if plot_type is not None:
        if plot_type == 'roc':
            metric_x='fpr'
            metric_y='tpr'
            leg_loc = 'lower right'
        elif plot_type == 'pr':
            metric_x='tpr'
            metric_y='ppv'
            leg_loc = 'upper right'
            label_dict.update({'tpr': 'Recall (%)'})
            del plot_spec['yticks']
            del plot_spec['ylim']
        elif plot_type == 'nne': 
            metric_x='tpr'
            metric_y='ppv-1'
            leg_loc = 'upper left'
            del plot_spec['yticks']
            del plot_spec['ylim']
        elif plot_type == 'npp': # number of predicted positive per 100
            metric_x='tpr'
            metric_y='ppr'
            leg_loc = 'upper left'
        elif plot_type == 'fpr':
            metric_x='thresholds'
            metric_y='fpr'
            leg_loc = 'upper right'
            label_dict.update({'fpr': 'False Positive Rate (%)'})
        elif plot_type == 'fnr':
            metric_x='thresholds'
            metric_y='fnr'
            leg_loc = 'upper left'
            label_dict.update({'fnr': 'False Negative Rate (%)'})
        elif plot_type == 'tpr':
            metric_x='thresholds'
            metric_y='tpr'
            leg_loc = 'upper right'
            label_dict.update({'tpr': 'True Positive Rate (%)'})
        elif plot_type == 'tnr':
            metric_x='thresholds'
            metric_y='tnr'
            leg_loc = 'upper left'
            label_dict.update({'tnr': 'True Negative Rate (%)'})
        else:
            raise NotImplementedError('Invalid plot_type')
    else:
        raise Exception("plot_type is None")
        
    print("..................Plot Type Initialization..............")
    
    #print(Y_validation, Y_validation_score, savePath)
    
    label_dict.update(label_dict_custom)
    plot_spec.update(plot_spec_custom)

    # import pdb; pdb.set_trace()
    
    if group_col is None:
        group_col = pd.Series(np.zeros(len(Y_validation)))
    assert isinstance(group_col, pd.Series), 'group_col has to be pd.Series object'
    
    print(group_col)
    
    plt.figure(figsize=(8 ,8), dpi= 300, facecolor='w', edgecolor='k')
    n_group = len(group_col.unique())  
    
    print("..................Plot Initialization..............")
    
    for _group in group_col.unique():
        if _group not in group_white_list:
            continue
        _Y_validation = Y_validation[group_col==_group]
        _Y_validation_score = Y_validation_score[group_col==_group]
        
        metric_tup = get_ROC_precisionRecall(_Y_validation,_Y_validation_score,CI_flag=CI_flag)
        metric_dict = dict(
            zip(
                ('roc_auc', 
                 'fpr', 
                 'tpr', 
                 'thresholds', 
                 'ppv', 
                 'npv', 
                 'precision_recall_auc',
                 'auc_CI_lower', 
                 'auc_CI_upper', 
                 'Y_score', 
                 'Y_true', 
                 'Pr_positive', 
                 'Pr_negative', 
                 'ppr'
                ), 
                metric_tup
            )
        )            
        metric_dict['ppv-1'] = 1/metric_dict['ppv']
        metric_dict['tnr'] = 1 - metric_dict['fpr']
        metric_dict['fnr'] = 1 - metric_dict['tpr']
        
        print("..................Drawing Curves..............", _group)

        '''
        # Serena 052422 racial bias analysis
        if _group == 'SEX_F' and plot_type == 'fnr':
            cut90 = sorted(metric_dict['Y_score'])[int(0.9*len(metric_dict['Y_score']))]
            cut95 = sorted(metric_dict['Y_score'])[int(0.95*len(metric_dict['Y_score']))]
            
            print("..................ERRORs..............", _group)
            fnr_dict = {'sen90': next((fnr for fnr, x in zip(metric_dict['fnr'], metric_dict['tpr']) if x >= 0.9)),\
                        'youden': metric_dict['fnr'][np.argmax(metric_dict['tpr']+(1-metric_dict['fpr']))],\
                        'cut90': next((fnr for fnr, x in zip(metric_dict['fnr'], metric_dict['thresholds']) if x < cut90)),\
                        'cut95': next((fnr for fnr, x in zip(metric_dict['fnr'], metric_dict['thresholds']) if x < cut95))}
            
        elif _group == 'SEX_M' and plot_type == 'fnr':
            #pdb.set_trace()
            print("..................ERRORs..............", _group)
            
            fnr_dict_black = {k: get_idx(metric_dict['fnr'], v, metric_dict['thresholds']) for k,v in fnr_dict.items()}
        '''
        
        print("..................Plot Initialization..............", _group)
        
        _X = metric_dict[metric_x] * (100 if metric_x in scale_list else 1) 
        _Y = metric_dict[metric_y] * (100 if metric_y in scale_list else 1) 
        
        _label = ', '.join(([_group] if _group else []) + (['C statistic = %.3f (%.3f, %.3f)'%(metric_dict['roc_auc'],metric_dict['auc_CI_lower'],metric_dict['auc_CI_upper']) if CI_flag else\
                                                            'C statistic = %.3f'%(metric_dict['roc_auc'])] if plot_type=='roc' else []))
        if _group in group_color:
            plt.plot(_X, _Y, linewidth=4.0, label=_label, color=group_color[_group])
        else:
            plt.plot(_X, _Y, linewidth=4.0, label=_label)
        
    for line in add_lines:
        line_copy = copy.deepcopy(line)
        args = line_copy.pop('args')
        plt.plot(*args,**line_copy)
        
    fontsize = plot_spec['fontsize']
    if 'xticks' in plot_spec:
        plt.xticks(plot_spec['xticks'])
    if 'yticks' in plot_spec:
        plt.yticks(plot_spec['yticks'])
    if 'xlim' in plot_spec:
        plt.xlim(plot_spec['xlim'])
    if 'ylim' in plot_spec:
        plt.ylim(plot_spec['ylim'])
    plt.grid()
    plt.xlabel(label_dict[metric_x], fontsize=fontsize)
    plt.ylabel(label_dict[metric_y], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)        
    plt.yticks(fontsize=fontsize)
    if title is not None:
        plt.title(title)
    plt.legend(loc=leg_loc, fontsize=fontsize)
    #plt.savefig('.'.join(["{}_{}".format(savePath.split('.')[0],plot_type),savePath.split('.')[1]]))    

    plt.savefig(savePath, dpi=300, bbox_inches='tight')