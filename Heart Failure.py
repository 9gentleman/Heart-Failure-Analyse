import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import norm
from scipy.stats import chi2
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import spearmanr
import statsmodels.api as sm

df = pd.read_excel(r'C:\Users\Can\Desktop\Project of Ist 2\cardio_train.xlsx')

selected_population_variables = df[['id', 'weight', 'ap_hi', 'ap_lo']]


# Fixing the outlier values in population.
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


def hypothesis_test(paramY, param_pop, param_lambda=0):
    # Kitle varyansı biliniyor
    n = len(paramY)
    standart_deviation_population = math.sqrt(param_pop.var())
    standart_error = standart_deviation_population/math.sqrt(n)
    sample_mean = paramY.mean()
    alpha = 0.1
    z_score = (sample_mean-param_lambda)/standart_error
    z_table = norm.ppf(1-(alpha/2))
    print(
        f'sdp {standart_deviation_population}, se {standart_error}, sm{sample_mean}, n {n}')
    if z_score >= z_table:
        print(f'z score: {z_score} >= z table:{z_table} olduğundan HO hipotezi Reddedilir.\n Kitle ortalamasi {param_lambda} dan farkli bir değer için.\n%10 anlamlilik düzeyinde tahmin yapilir.')
    else:
        print(f'z score: {z_score} < z table:{z_table} olduğundan HO hipotezi Reddedilemez.\n Kitle ortalamasi {param_lambda} için.\n%10 anlamlilik düzeyinde tahmin yapilir.')


def corr_matrix_plt(param_df):
    corr_matrix = param_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelasyon Matrisi')
    plt.show()


def normality_test(param):
    """
        Info: param must be data frame.
    """
    # Shapiro-Wilk testi
    stat, p_value = shapiro(param)
    print("Test İstatistiği:", stat)
    print("p-değeri:", p_value)

    alpha = 0.05
    if p_value > alpha:
        print("Örneklem normal dağilimdan gelmektedir (H0 reddedilemez)")
    else:
        print("Örneklem normal dağilimdan gelmemektedir (H0 reddedilir)")


def corr_test(paramX, paramY):
    spearman_corr, p_value = spearmanr(paramX, paramY)
    print("Spearman katsayisi:", spearman_corr)
    print("P değeri:", p_value)
    if 0 <= spearman_corr < 0.10:
        print('İlişki Yoktur')
    if 0.10 <= spearman_corr < 0.20:
        print('Zayif ilişki.')
    if 0.20 <= spearman_corr < 0.40:
        print('Çok Zayif ilişki.')
    if 0.40 <= spearman_corr < 0.60:
        print('Orta Düzey ilişki.')
    if 0.60 <= spearman_corr < 0.80:
        print('Yüksek ilişki.')
    if 0.80 <= spearman_corr < 0.90:
        print('Çok Yüksek ilişki.')
    if 0.90 <= spearman_corr <= 1:
        print('Mükemmel ilişki.')


def reg_model(paramX, paramY):
    """
        Info: ParamX = Independent
              ParamY = dependent
    """
    X = paramX
    Y = paramY
    # Sabit terimi ekleyerek X'i yeniden şekillendirme
    X = sm.add_constant(X)
    # Modeli oluşturma
    model = sm.OLS(Y, X).fit()
    print(model.summary())
