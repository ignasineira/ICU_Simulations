
from scipy import stats 


df2,time_diff_columns=data_variant()
df1,time_diff_columns=data()



for i, (group_name, group_df) in enumerate(df1.groupby(["Grupo de edad"])):
    # t test using scipy
    print(group_name)
    x1 = df1.loc[df1['Grupo de edad'] == group_name, 'T. EGRESO UCI desde INGRESO UCI'].to_numpy()
    x2 = df2.loc[df2['Grupo de edad'] == group_name, 'T. egreso UCI desde ingreso UCI'].to_numpy()
    x2=x2[~np.isnan(x2)]
    # Calculate the mean and standard error
    x1_bar, x2_bar = np.mean(x1), np.mean(x2)
    
    print("promedio: {},{}".format(round(x1_bar,3), round(x2_bar,3)))
    n1, n2 = len(x1), len(x2)
    var_x1, var_x2= np.var(x1, ddof=1), np.var(x2, ddof=1)
    
    # pooled sample variance
    pool_var = ( ((n1-1)*var_x1) + ((n2-1)*var_x2) ) / (n1+n2-2)
    
    # standard error
    std_error = np.sqrt(pool_var * (1.0 / n1 + 1.0 / n2))
    
    # calculate t statistics
    t = abs(x1_bar - x2_bar) / std_error
    print("calculate t statistics: {}".format(round(t,4)))
    #print(t)
    # two-tailed critical value at alpha = 0.05
    # q is lower tail probability and df is the degrees of freedom
    degrees_freedom=n1+n2-1
    t_crit = stats.t.ppf(q=0.975, df=degrees_freedom)
    print("calculate t critical: {}".format(round(t_crit,4)))
    # get two-tailed p value
    p = 2*(1-stats.t.cdf(x=t, df=degrees_freedom))
    print("calculate p value: {}".format(round(p,4)))
    #print(st.ttest_ind(a=a, b=b, equal_var=True))
    
for i, (group_name, group_df) in enumerate(df1.groupby(["Grupo de edad"])):
    # t test using scipy
    print(group_name)
    x1 = df1.loc[df1['Grupo de edad'] == group_name, 'T. INGRESO UCI desde INICIO DE LOS SINTOMAS'].to_numpy()
    x2 = df2.loc[df2['Grupo de edad'] == group_name, 'T. ingreso UCI desde inicio sintomas'].to_numpy()
    x2=x2[~np.isnan(x2)]
     # Calculate the mean and standard error
    x1_bar, x2_bar = np.mean(x1), np.mean(x2)
    
    print("promedio: {},{}".format(round(x1_bar,3), round(x2_bar,3)))
    n1, n2 = len(x1), len(x2)
    var_x1, var_x2= np.var(x1, ddof=1), np.var(x2, ddof=1)
    
    # pooled sample variance
    pool_var = ( ((n1-1)*var_x1) + ((n2-1)*var_x2) ) / (n1+n2-2)
    
    # standard error
    std_error = np.sqrt(pool_var * (1.0 / n1 + 1.0 / n2))
    
    # calculate t statistics
    t = abs(x1_bar - x2_bar) / std_error
    print("calculate t statistics: {}".format(round(t,4)))
    #print(t)
    # two-tailed critical value at alpha = 0.05
    # q is lower tail probability and df is the degrees of freedom
    degrees_freedom=n1+n2-1
    t_crit = stats.t.ppf(q=0.975, df=degrees_freedom)
    print("calculate t critical: {}".format(round(t_crit,4)))
    # get two-tailed p value
    p = 2*(1-stats.t.cdf(x=t, df=degrees_freedom))
    print("calculate p value: {}".format(round(p,4)))
    #print(st.ttest_ind(a=a, b=b, equal_var=True))
    #print(st.ttest_ind(a=a, b=b, equal_var=True))
#Interpretation
#The p value obtained from the t-test is significant (p < 0.05),
# and therefore, we conclude that the yield of x1 is significantly different than x2
