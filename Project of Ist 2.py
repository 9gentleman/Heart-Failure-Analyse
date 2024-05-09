import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import norm
from scipy.stats import chi2

df = pd.read_excel(r'C:\Users\Can\Desktop\Project of Ist 2\cardio_train.xlsx')

selected_population_variables = df[["id", "weight", "ap_hi", "ap_lo"]]


# Fixing the outlier values in pop
selected_population_variables.loc[(selected_population_variables['ap_lo'] > 120) | (
    selected_population_variables['ap_lo'] < 60), 'ap_lo'] = 80

selected_population_variables.loc[(selected_population_variables['ap_hi'] > 170) | (
    selected_population_variables['ap_hi'] < 75), 'ap_hi'] = 105

selected_population_variables.loc[selected_population_variables['weight']
                                  > 170, 'weight'] = 85


def outler_detec(val, name='', level=0):
    """
        Info: sample mean +- level*sample standart deviation
    """
    formula_outlier_upper = val.mean() + (level * math.sqrt(val.var()))
    formula_outlier_lower = val.mean() - (level * math.sqrt(val.var()))
    formula_outlier = (formula_outlier_lower, formula_outlier_upper)
    print(f'{name} : {formula_outlier}')


def check_outlier(data, max_lim=0, min_lim=0, name_data='name of data'):
    outlier_count = 0
    for inner in data.values:
        if (inner > max_lim or inner < min_lim):
            outlier_count += 1
        else:
            pass
    if outlier_count == 0:
        print(f'{name_data} için Aykiri Değer Yoktur.')
    else:
        print(f'{name_data} için Aykiri Değer ya da Değerler vardir.')
        print(f'{name_data} için Aykiri Değer sayisi: {outlier_count}')


def sample_size_cal(pop):
    """
        Info: This function calculate the size of population.
        Parameter: Populaiton
    """
    # CONFIDANCE_LEVEL = 0.95
    variance_pop = pop.var()
    pop_size = len(pop)
    DEVIATION = 10
    Z_TABLE = 2
    P = 0.5
    Q = 1-P
    sample_size = ((pop_size)*(Z_TABLE**2)*(variance_pop)) / \
        ((DEVIATION**2)*(pop_size-1)+(Z_TABLE**2)*(variance_pop))
    rounded_sample_size = round(sample_size)
    return rounded_sample_size


# Sample
pop_size = len(df)
sampleSize = 1000

sample_index = [i for i in range(0, pop_size, 70)]
missing_data = sampleSize - len(sample_index)

if missing_data > 0:
    missing_data = random.sample(range(pop_size), missing_data)
    sample_index.extend(missing_data)

sample = selected_population_variables.iloc[sample_index]

selected_sample_variables = sample[["id", "weight", "ap_hi", "ap_lo"]]


df_ap_lo = selected_sample_variables['ap_lo']
df_ap_hi = selected_sample_variables['ap_hi']
df_weight = selected_sample_variables['weight']


def analyse_descriptive_statistic(param, param_name):
    """
        Info: This function basic analyse to data. Such as(mean, variance, median, quantiles, standart deviation)
        Parameters: param = Dataset, param_name = name of dataset 
    """
    param_mean = param.mean()
    param_var = param.var()
    param_median = param.median()
    # param_mode = param.mode()
    param_quantiles = param.quantile()
    param_standart_deviation = math.sqrt(param.var())
    print(f"""
          {param_name} mean is : {param_mean}
          {param_name} variance is : {param_var}
          {param_name} median is : {param_median}
          {param_name} quantiles is : {param_quantiles}
          {param_name} standart deviation is : {param_standart_deviation}""", sep="")


def visualization_data(data1, data2, graph):
    """
        Info: this function plot graphs
        x: data1
        y: data2
        graphs:
            graphs = 1 histogram
            graphs = 2 Trend
            graphs = 3 Curve
    """
    plt.xlabel = data1
    plt.ylabel = data2
    if graph == 1:
        plt.title('Graph Of Histogram')
        plt.hist(data1, bins=15, color="skyblue")
        plt.grid(True)
        plt.show()
    if graph == 2:
        plt.title('Graph Of Trend')
        sns.regplot(x=data1, y=data2, ci=None)
        plt.grid(True)
        plt.show()
    if graph == 3:
        plt.title('Graph Of Curve')
        plt.plot(data1, data2,
                 marker='o', color='green', linestyle='-')
        plt.grid(True)
        plt.show()
    else:
        pass


def confidence_interval_mean(param_sample, param_population):
    alpha = 0.05
    sample_size = len(param_sample)
    z_value = norm.ppf(1-(alpha/2))  # table score
    samp_mean = param_sample.mean()
    pop_var = param_population.var()
    pop_ss = math.sqrt(pop_var)
    low_limit = samp_mean - z_value * (pop_ss / math.sqrt(sample_size))
    upper_limit = samp_mean + z_value * (pop_ss / math.sqrt(sample_size))
    confidance_range = low_limit, upper_limit
    return confidance_range


def confidence_interval_var(param_sample):
    alpha = 0.1
    n = len(param_sample)
    degree_of_freedom = n-1
    samp_var = param_sample.var()
    critical_value1 = chi2.ppf(1-(alpha/2), degree_of_freedom)
    critical_value2 = chi2.ppf(alpha/2, degree_of_freedom)
    low_limit = ((degree_of_freedom)*(samp_var))/(critical_value1)
    upper_limit = ((degree_of_freedom)*(samp_var))/(critical_value2)
    confidance_range = low_limit, upper_limit
    return confidance_range


def hypothesis_test(paramY):
    # Ho : 𝜇 = 120
    # Hs : 𝜇 ≠ 120
    # ap hi için yapalım 120 olup olmadığını bakalım, Kitle varyansı biliniyor
    selected_population_variables['ap_hi'].var()
    alpha = 0.1
    pass


# Y değişkenini kullanarak belirlediğiniz bir 𝜇0 değeri için 𝐻0:𝜇=𝜇0 hipotezini 𝐻S:𝜇≠𝜇0 hipotezine karşı %10 anlamlılık düzeyinde test ediniz ve yorumlayınız.

# XX ve Y değişkenleri arasında Pearson, Spearman ve Kendall’ın korelasyon katsayılarından en uygun olanını hesaplayarak yorumlayınız.

# X ve Y değişkenleri arasında basit doğrusal regresyon denklemini kurunuz, denklemin anlamlılığını % 5 anlamlılık düzeyinde test ediniz ve yorumlayınız.
