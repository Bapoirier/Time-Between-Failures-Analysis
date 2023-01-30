import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines import WeibullAFTFitter

# Para ajustar los modelos
# =============================================================================================================================
def wb_AFT_reg(data,time_variable,residuals_list,ndecimals,print_results):

    weibull_aft = WeibullAFTFitter()  # Call the fitter from lifelines
    weibull_aft.fit(data, duration_col=time_variable) # fit the model
    
    # Get the resulting parameters
    res = pd.DataFrame(weibull_aft.params_).reset_index() 
    
    #parameters associated with the Weibull scale parameter λ
    res_lambda = res[res.param=='lambda_'].rename(columns={0:'value'}) 
    res_beta = res[res.param=='rho_'].rename(columns={0:'value'})
    print('Estimated scale parameter λ = ', np.exp(res_lambda.value.sum()))
    
    # Save the residuals
    model_res = []
    for i in data[[time_variable]].values:
        rc = np.exp((np.log(i[0]) - res_lambda.value.sum())/(1/np.exp(weibull_aft.params_.rho_.Intercept)))
        model_res.append(rc)
    residuals_list.append(model_res)
    
    # Print summary
    if print_results == True:
        weibull_aft.print_summary(decimals=ndecimals)
    return (res_lambda,res_beta)
# =============================================================================================================================
    
    
    
# Para comparar los residuos
# =============================================================================================================================
def compare_residuals(ax,residual,model_name):

    naf = NelsonAalenFitter() #Call the fitter from lifelines
    naf.fit(residual) 
    naf.plot_cumulative_hazard(ax=ax, label=model_name)

    # Get the cumulative hazard values from the Nelson-Aalen estimate
    aux = naf.cumulative_hazard_.reset_index()
    
    d = 0
    dmodel = []
    for i,row in aux.iterrows():
        if i!=0:
            dmodel.append(d)
            d += np.abs(row[1]-row[0])*(aux.loc[i,'timeline'] - aux.loc[i-1,'timeline'] )
    
    print(model_name + ' d =',d)
    return dmodel
# =============================================================================================================================

def evaluate_survivals(T,thetas,beta,df,caex_list,model,fitted_model):
    norm_df = (df - df.min()) / (df.max()-df.min())
    rows  = []
    if model=='weibull':
        for i in range(len(thetas[:-1])):
                covariate = thetas.loc[i,'covariate']
                theta = thetas.loc[i,'value']
                rows.append(norm_df[covariate]*theta)
        X = pd.DataFrame(np.array(rows)).sum() + thetas.loc[len(thetas)-1,'value']
        
    elif model=='cox':
        for i in range(len(thetas)):
                covariate = thetas.loc[i,'covariate']
                theta = thetas.loc[i,'coef']
                rows.append(norm_df[covariate]*theta)
        X = pd.DataFrame(np.array(rows)).sum()
        
    lsurv    = []
    lnotsurv = []
    probs    = []
    for i in range(len(X)): 
        if model=='weibull':
            p_surv = np.exp(-(T/np.exp(X[i]))**np.exp(beta.reset_index().loc[0,'value']))
            p = 0.75
        elif model=='cox':
            p_surv = np.exp(-np.exp(X[i])*fitted_model.baseline_cumulative_hazard_)[:T+1]
            p_surv = p_surv.reset_index().loc[len(p_surv)-1,'baseline cumulative hazard']
        probs.append(p_surv)
    j = 0        
    probs_mean = np.mean(probs)
    for p in probs:
        if p>probs_mean:
            lsurv.append(caex_list[j])
        else:
            lnotsurv.append(caex_list[j])
        j+=1
        
    return lsurv , lnotsurv, probs
# ===============================================================================

# To calculate confusion matrices
# ===============================================================================
def conf_matrix(act_true,act_false,pred_true,pred_false):
    tp,fp,fn,tn = 0,0,0,0
    for i in pred_true:
        if i in act_true:
            tp+=1
        if i in act_false:
            fp+=1
    for i in pred_false:
        if i in act_true:
            fn+=1
        if i in act_false:
            tn+=1
    return np.array([[tp,fn],[fp,tn]])
# ===============================================================================

# Metricas asocidas a la matriz de confución
# ===============================================================================
def conf_matrix_metrics(cm):
    acc = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0])
    rec = cm[0,0] / (cm[0,0] + cm[0,1] )
    pre = cm[0,0] / (cm[0,0] + cm[1,0] )
    f1s = 2*cm[0,0]/(2*cm[0,0] +cm[1,0]+cm[0,1])
    return [acc,rec,pre,f1s]
# ===============================================================================