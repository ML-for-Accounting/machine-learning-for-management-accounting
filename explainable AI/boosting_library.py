# Collection of functions for boosting based nonlinear regression. Focus on sig. analysis
# Prepare data
# Winsorize data
# cv_boost
# shap_table
# return_levels

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import numpy as np

plt.style.use('bmh')

def prepare_data(full_df, variable):
	"""Remove the nans based on the y-variable. Returns the prepared table."""
	full_df = full_df.loc[~full_df[variable].isna()]
	return full_df.reset_index(drop=True)

def winsorize_data(input_df,limits=0.01):
	"""Winsorize data"""
	return input_df.clip(lower=input_df.quantile(limits), upper=input_df.quantile(1-limits), axis = 1)

def cv_boost(x_df, y_df, param, boost_rounds = 1000, plot_low = 50,cv_metric = 'mae',esr=10):
	"""Basic cross-validation routine with common parameters. Returns the optimal number of boosting rounds AND parameters"""
	dtrain = xgb.DMatrix(x_df, label=y_df, nthread = -1)
	temp = xgb.cv(param,dtrain,num_boost_round=boost_rounds,nfold=5,seed=None,metrics=cv_metric,early_stopping_rounds=esr)
	plot_high= len(temp)
	fig, axs = plt.subplots(1,2,figsize=(12,6),squeeze=True)
	axs[0].plot(temp['test-'+cv_metric+'-mean'][plot_low:plot_high],'r--')
	axs[1].plot(temp['train-'+cv_metric+'-mean'][plot_low:plot_high],'r--')
	plt.show()
	return plot_high,param

# Calculates the mean of absolute SHAP values. Returns a dataframe with the names of features and the mean values PLUS the original SHAP_table 
def shap_table(x_df, y_df, param, b_rounds):
    dtrain = xgb.DMatrix(x_df, label=y_df, nthread = -1)
    bst = xgb.train(param,dtrain,num_boost_round=b_rounds)
    explainerXGB = shap.TreeExplainer(bst)
    shap_values_np = explainerXGB.shap_values(x_df,y_df)
    shaps = np.mean(abs(shap_values_np), axis = 0)
    names = bst.feature_names
    apu_df = pd.DataFrame()
    apu_df['names'] = names
    apu_df['shaps'] = shaps
    return apu_df, shap_values_np

def random_sampler(full_df,n_values=1000):
	"""Create fully random sample_df with replacement."""
	sample_df = full_df.sample(n=n_values,replace=True)
	sample_df.reset_index(inplace=True,drop=True)
	return sample_df
    
def block_sampler(full_df, variable,n_values=1000):
	"""Create sample_df using block sampling. Uses random samples of "block" variable (with replacement)."""
	sample_df = pd.DataFrame()
	uniques_df = pd.DataFrame(full_df[variable].unique())
	fraction = n_values/len(full_df)+0.04
	for value in uniques_df.sample(frac=fraction,replace=True)[0]:
		sample_df = sample_df.append(full_df[full_df[variable] == value],ignore_index=True)
	sample_df.reset_index(inplace=True,drop=True)
	return sample_df.iloc[0:n_values]

# Average levels of percentage intervals. REMEMBER TO RESET X_df index! Returns the bin limits and the average value for each bin.
def return_adaptive_levels(SHAP_values_df,x_df,variable,perc_interval):
    bins = []
    ind = 0.
    while ind < 1.001:
        bins.append(x_df[variable].quantile(ind))
        ind+=perc_interval
    test,bins1 = pd.cut(x_df[variable],bins,include_lowest=True,duplicates='drop',retbins=True)
    means = SHAP_values_df[variable].groupby(test).mean()
    lows = bins1[:-1]
    highs = bins1[1:]
    return means,lows,highs

# Average levels of percentage intervals. REMEMBER TO RESET X_df index! Returns the bin limits and the average value for each bin.
def return_constant_levels(SHAP_values_df,x_df,variable,constant_interval):
    bins = np.linspace(x_df[variable].min(),x_df[variable].max(),(x_df[variable].max()-x_df[variable].min())*constant_interval)
    test,bins1 = pd.cut(x_df[variable],bins,include_lowest=True,duplicates='drop',retbins=True)
    means = SHAP_values_df[variable].groupby(test).mean()
    lows = bins1[:-1]
    highs = bins1[1:]
    return means,lows,highs

# Average levels of FIXED intervals. REMEMBER TO RESET X_df index! Returns the bin limits and the average value for each bin.
def return_fixed_levels(SHAP_values_df,x_df,variable,no_bins):
    test,bins1 = pd.cut(x_df[variable],no_bins,include_lowest=True,duplicates='drop',retbins=True)
    means = SHAP_values_df[variable].groupby(test).mean()
    lows = bins1[:-1]
    highs = bins1[1:]
    return means,lows,highs

# Calcultes shap values with automatic optimisation using cross-validation.
def boot_shap(sample_df,x_variables,y_variable,param,metrics,esr,b_rounds):
    y_df = sample_df[y_variable]
    x_df = sample_df[x_variables]
    dtrain = xgb.DMatrix(x_df, label=y_df, nthread = -1)
    if not b_rounds:
        cv_results = xgb.cv(param,dtrain,num_boost_round=1000,nfold=5,early_stopping_rounds=esr,seed=None,metrics=metrics)
        b_rounds = len(cv_results)
        print("Optimal number of trees: " + str(b_rounds), end = '\r', flush=True)
    bst = xgb.train(param,dtrain,num_boost_round=b_rounds)
    explainerXGB = shap.TreeExplainer(bst)
    shap_values_np = explainerXGB.shap_values(x_df,y_df,check_additivity=False)
    shap_values_df = pd.DataFrame(shap_values_np,columns = x_df.columns)
    return shap_values_df

# Bootstrap average levels for Fixed bins. If you are using block-sampling, the sample-variation makes it difficult to use adaptive bins.
def bootstrap_fixed_bins(bootstrap_shaps_df,bootstrap_samples_df,nobs,variable, no_bins ,conf_limit=0.05):
    rounds = int(len(bootstrap_shaps_df)/nobs)
    samples_levels_df = pd.DataFrame()
    for ind in range(rounds):
        means,lows,highs = return_fixed_levels(bootstrap_shaps_df.iloc[ind*nobs:(ind+1)*nobs],bootstrap_samples_df.iloc[ind*nobs:(ind+1)*nobs],variable,no_bins)
        samples_levels_df['Sample ' + str(ind)] = means
    mean = samples_levels_df.mean(axis=1)
    variance = samples_levels_df.var(axis=1)
    low_limit = samples_levels_df.quantile(conf_limit,axis=1)
    hi_limit = samples_levels_df.quantile(1-conf_limit,axis=1)
    samples_levels_df['Overall mean'] = mean
    samples_levels_df['Variance'] = variance
    samples_levels_df[str(conf_limit*100) + '% line'] = low_limit
    samples_levels_df[str((1-conf_limit)*100) + '% line'] = hi_limit
    samples_levels_df['Lows'] = lows
    samples_levels_df['Highs'] = highs
    return samples_levels_df

# Calcultes bootstrap. Returns x-values dataframe and shap-dataframe.
def calculate_bootstrap(original_df,x_variables,y_variable,param,block_variable,n_values,metrics,num_trees=False,boot_rounds=50, esr=50):
    bootstrap_shaps_df = pd.DataFrame()
    bootstrap_samples_df = pd.DataFrame()
    for i in range(boot_rounds):
        sample_df = block_sampler(original_df,block_variable,n_values=n_values)
#        sample_df = random_sampler(original_df,n_values=n_values)
        bootstrap_samples_df = bootstrap_samples_df.append(sample_df[x_variables])
        print('Round: ' + str(i), end = '\r',flush=True)
        bootstrap_shaps_df = bootstrap_shaps_df.append(boot_shap(sample_df,x_variables,y_variable,param, esr = esr,metrics=metrics,b_rounds=num_trees))
    return bootstrap_samples_df,bootstrap_shaps_df

# Plot ALL bootstrap shap values with a mean value and conf intervals
def plot_bootstrarp_shaps(bootstrap_samples_df,bootstrap_shaps_df,x_vars, nobs, no_bins, plot_arr,
                          c_limit,c_bool, name_prefix, alpha, y_limits, defined_limits = False, figsize=(20,20), save_fig = True,hspace=0.2,show_obs=True):
    fig, axs = plt.subplots(plot_arr[0], plot_arr[1],figsize=figsize,squeeze=True)
    for variable,ax in zip(x_vars,axs.flat):
        if defined_limits:
            cheat_bins = np.linspace(defined_limits[0],defined_limits[1],no_bins)
        else:
            half_bin_length = (bootstrap_samples_df[variable].max()-bootstrap_samples_df[variable].min())/(2*no_bins)
            cheat_bins = np.linspace(bootstrap_samples_df[variable].min()-half_bin_length,bootstrap_samples_df[variable].max()+half_bin_length,no_bins)
        samples_levels_df = bootstrap_fixed_bins(bootstrap_shaps_df,bootstrap_samples_df,nobs,variable,conf_limit=c_limit,no_bins=cheat_bins)
        samples_levels_df.dropna(inplace = True,subset=['Overall mean'])
        ax.plot((samples_levels_df['Lows']+samples_levels_df['Highs'])/2,samples_levels_df['Overall mean'],color='black',linewidth=2)
        if c_bool:
            ax.plot((samples_levels_df['Lows']+samples_levels_df['Highs'])/2,samples_levels_df[str(c_limit*100) + '% line'],'k--',linewidth=2)
            ax.plot((samples_levels_df['Lows']+samples_levels_df['Highs'])/2,samples_levels_df[str((1-c_limit)*100) + '% line'],'k--',linewidth=2)
        if show_obs:
            ax.scatter(bootstrap_samples_df[variable],bootstrap_shaps_df[variable],s=1,alpha=alpha,color='gray')
        if y_limits:
            ax.set_ylim(y_limits)
        ax.set_xlabel(variable)
        ax.set_ylabel('SHAP value for\n' + variable)
    plt.subplots_adjust(hspace=hspace)
    if save_fig:
        plt.savefig(name_prefix + '_bootstr_shaps_trend.png',dpi=500,facecolor='w')

def build_sig_table(bootstrap_shaps_df, bootstrap_samples_df, variables, nobs, no_bins, conf_limit,reset_bins,show_limits,include_conf_limits=True, just_means=False):
    results_df = pd.DataFrame()
    for variable in variables:
        fixed_bins_df = bootstrap_fixed_bins(bootstrap_shaps_df, bootstrap_samples_df, nobs, variable, conf_limit=conf_limit, no_bins=no_bins)
        prod_limits = fixed_bins_df[str(conf_limit*100) + '% line']*fixed_bins_df[str((1-conf_limit)*100) + '% line']
        if just_means:
            fixed_bins_df.reset_index(inplace= True)
            results_df[variable] = fixed_bins_df['Overall mean']
            continue
        if ~prod_limits.isna().values.any():
            fixed_bins_df['sign'] = np.sign(prod_limits).astype(int)
            fixed_bins_df['sign'].replace({1:'*',-1:''},inplace=True)
        else:
            fixed_bins_df['sign'] = 'o'
        if reset_bins:
            fixed_bins_df.reset_index(inplace= True)
        if include_conf_limits:
            results_df[variable] = fixed_bins_df['Overall mean'].round(4).astype(str) + ' (' + fixed_bins_df[str(conf_limit*100) + '% line'].round(4).astype(str) + ', ' + fixed_bins_df[str((1-conf_limit)*100) + '% line'].round(4).astype(str) + ')' + fixed_bins_df['sign']
        else:
            results_df[variable] = fixed_bins_df['Overall mean'].round(4).astype(str) + fixed_bins_df['sign']
        temp = []
        _,bins = pd.cut(bootstrap_samples_df[variable],no_bins,retbins=True)
        if show_limits:
            for i in range(len(bins)-1):
                temp.append(str(round(bins[i],1))+' - '+str(round(bins[i+1])))
            results_df[variable+' - limits']= temp
    return results_df.set_index(results_df.index.rename('Variable')).transpose()
    
def plot_mean_sig(bootstrap_shaps_df, bootstrap_samples_df, variables, nobs, no_bins, conf_limit, plot_arr,name_prefix):
    fig, axs = plt.subplots(plot_arr[0], plot_arr[1],figsize=(20,20),squeeze=True)
    for variable,ax in zip(variables,axs.flat):
        test_df = bootstrap_fixed_bins(bootstrap_shaps_df,bootstrap_samples_df,nobs,variable, conf_limit=conf_limit, no_bins=no_bins)
        ax.hlines(test_df['Overall mean'],test_df['Lows'],test_df['Highs'], linewidth=1)
        ax.hlines(test_df['5.0% line'],test_df['Lows'],test_df['Highs'],color='black',linewidth=1, linestyles='dashed')
        ax.hlines(test_df['95.0% line'],test_df['Lows'],test_df['Highs'],color='black',linewidth=1, linestyles='dashed')
        ax.set_title(variable)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(name_prefix + '_mean_sig_plot.png',dpi=500)
    
"""def build_importance_table(bootstrap_shaps_df, bootstrap_samples_df, variables, nobs, no_bins, importance_conf_limit):
    results_df = pd.DataFrame()
    boot_samples = int(len(bootstrap_shaps_df)/nobs)
    name = []
    mean = []
    lo_limit = []
    hi_limit = []
    for variable in variables:
        fixed_bins_df = bootstrap_fixed_bins(bootstrap_shaps_df, bootstrap_samples_df, nobs, variable, no_bins=no_bins)
        sample_means = fixed_bins_df.abs().mean()[0:boot_samples]
        name.append(variable)
        mean.append(sample_means.mean())
        lo_limit.append(sample_means.quantile(importance_conf_limit))
        hi_limit.append(sample_means.quantile(1-importance_conf_limit))
    results_df['Name'] = name
    results_df['Mean'] = mean
    results_df['Low limit'] = lo_limit
    results_df['High limit'] = hi_limit
    return results_df"""

# THIS VERSION CALCULATES THE IMPORTANCE USING INTERVALS THAT HAVE EQUAL AMOUNT OF OBSERVATIONS
def build_importance_table(bootstrap_shaps_df, bootstrap_samples_df, variables, perc_interval, importance_conf_limit):
    results_df = pd.DataFrame()
    name = []
    mean = []
    for variable in variables:
        adap_bins,_,_ = return_adaptive_levels(bootstrap_shaps_df, bootstrap_samples_df, variable, perc_interval)
        name.append(variable)
        mean.append(np.mean(np.abs(adap_bins)))
    results_df['Name'] = name
    results_df['Mean'] = mean
    return results_df

def plot_bootstrap_hlines(bootstrap_shaps_df,bootstrap_samples_df,nobs,no_bins, name_prefix,plot_stats, alpha, plot_arr, save_fig = True,conf_limit=0.05):
    fig, axs = plt.subplots(plot_arr[0], plot_arr[1],figsize=(20,20),squeeze=True)
    rounds = int(len(bootstrap_shaps_df)/nobs)
    for variable,ax in zip(bootstrap_samples_df.columns,axs.flat):
        samples_levels_df = bootstrap_fixed_bins(bootstrap_shaps_df,bootstrap_samples_df,nobs,variable,no_bins=no_bins,conf_limit=conf_limit)
        for ind in range(rounds):
            ax.hlines(samples_levels_df['Sample ' + str(ind)],samples_levels_df['Lows'],samples_levels_df['Highs'],color='gray',alpha=alpha,linewidth=1)
        if plot_stats:
            ax.hlines(samples_levels_df['Overall mean'],samples_levels_df['Lows'],samples_levels_df['Highs'], linewidth=2, color = 'black')
            ax.hlines(samples_levels_df[str(conf_limit*100) + '% line'],samples_levels_df['Lows'],samples_levels_df['Highs'],color='black',linewidth=2, linestyles='dashed')
            ax.hlines(samples_levels_df[str((1-conf_limit)*100) + '% line'],samples_levels_df['Lows'],samples_levels_df['Highs'],color='black',linewidth=2, linestyles='dashed')
            ax.set_title(variable)
    plt.subplots_adjust(hspace=0.3)
    if save_fig:
        plt.savefig(name_prefix + '_bootstr_hlines.jpg',dpi=300,format='jpg')
        
# FEATURE IMPORTANCE FROM BIN DIFFERENCE
def difference_importance_table(bootstrap_shaps_df, bootstrap_samples_df, variables, nobs, no_bins,
                                conf_limit):
    results_df = pd.DataFrame()
    name = []
    mean_effect = []
    low_limit = []
    hi_limit = []
    for variable in variables:
        fixed_bins_df = bootstrap_fixed_bins(bootstrap_shaps_df, bootstrap_samples_df, nobs, variable,
                                             conf_limit=conf_limit, no_bins=no_bins)
        fixed_bins_df = fixed_bins_df.iloc[:,:-6]
        max_effect = []
        for column in fixed_bins_df.columns:
            max_effect.append(fixed_bins_df[column].max()-fixed_bins_df[column].min())
        name.append(variable)
        mean_effect.append(np.nanmean(max_effect))
        low_limit.append(np.nanquantile(max_effect,conf_limit))
        hi_limit.append(np.nanquantile(max_effect,1-conf_limit))
    results_df['Feature'] = name
    results_df['Effect difference'] = mean_effect
    results_df['Low limit'] = low_limit
    results_df['High limit'] = hi_limit
    return results_df

# FEATURE IMPORTANCE FROM percentage BINS
def percentage_difference_importance_table(bootstrap_shaps_df, bootstrap_samples_df, variables, nobs, perc_interval,
                                conf_limit):
    results_df = pd.DataFrame()
    name = []
    mean_effect = []
    low_limit = []
    hi_limit = []
    for variable in variables:
        rounds = int(len(bootstrap_shaps_df)/nobs)
        samples_levels_df = pd.DataFrame()
        max_effect = []
        for ind in range(rounds):
            means,lows,highs = return_adaptive_levels(bootstrap_shaps_df.iloc[ind*nobs:(ind+1)*nobs],
                                                   bootstrap_samples_df.iloc[ind*nobs:(ind+1)*nobs],
                                                   variable,perc_interval)
            max_effect.append(means.max()-means.min())
        name.append(variable)
        mean_effect.append(np.nanmean(max_effect))
        low_limit.append(np.nanquantile(max_effect,conf_limit))
        hi_limit.append(np.nanquantile(max_effect,1-conf_limit))
    results_df['Feature'] = name
    results_df['Effect difference'] = mean_effect
    results_df['Low limit'] = low_limit
    results_df['High limit'] = hi_limit
    return results_df