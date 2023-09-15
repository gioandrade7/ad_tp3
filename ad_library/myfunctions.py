import math
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import random

from functools import reduce


def freq_dist_discrete(values: list):
    v = sorted(values)
    n = len(v)
    #Como a variável é discreta quantitativa, teremos que gerar uma distribuição por intervalos de classes. 
    # k = 1 + (3.3 * math.log10(n))
    # k = int(round(k, 0))
    k = int(math.sqrt(n))

    amp_total = max(v) - min(v)
    h = math.ceil(amp_total / k)

    ls = min(v) + h
    classes = []
    ni = []
    for _ in range(k):
        classes.append(f'{ls-h} |- {ls}')
        ni.append(len([x for x in v if (ls - h) <= x < ls]))
        ls += h

    d = {
        'classe': classes,
        'ni': ni
    }
    df = pd.DataFrame(d)

    df['fi'] = df['ni'] / df['ni'].sum()
    df['100fi'] = df['fi'] * 100

    return df

def mean(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular média para uma lista vazia')
    # sum = reduce(lambda x,y: x + y, values)
    x = sum(values) / len(values)
    if decimals:
        return round(x, decimals)
    return x

def weighted_mean(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular média ponderada para uma lista vazia')
    sum_p = reduce(lambda x,y: x+y, [v*w for v,w in values])
    # sum_w = reduce(lambda x,y: x+y, weights)
    wx = sum_p / sum([v[1] for v in values])
    if decimals:
        return round(wx, decimals)
    return wx
    
def geometric_mean(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular média geométrica para uma lista vazia')
    n = len(values)
    g = reduce(lambda x,y: x*y, values)
    if (g < 0):
        raise ValueError('Não são permitidos valores negativos na média geométrica.')
    gx = math.exp(math.log(g) / n)
    if decimals:
        return round(gx, decimals)
    return gx

def harmonic_mean(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular média harmônica para uma lista vazia')
    k = len(values)
    h = sum([1./x for x in values])
    hx = k / h
    if decimals:
        return round(hx, decimals)
    return hx

def tax_mean(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular média de taxas para uma lista vazia')
    a = [v[1] for v in values]
    b = [v[0] for v in values]

    if all([bi==b[0] for bi in b]):
        tx = sum(a) / (len(values) * b[0])
    elif all([ai==a[0] for ai in a]):
        tx = len(values) / sum([bi/a[0] for bi in b])
    else:
        tx = (sum(a)/len(a)) / (sum(b)/len(b))
    
    if decimals:
        return round(tx, decimals)
    return tx

def median(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular mediana para uma lista vazia')
    length = len(values)
    v = sorted(values)
    
    if(length % 2 == 0):
        mid = length // 2
        md = (v[mid-1] + v[mid]) / 2.
    else:
        md = v[length // 2]

    if decimals:
        return round(md, decimals)
    return md

def mode(values: list):
    if len(values) == 0:
        raise ValueError('Não se pode calcular moda para uma lista vazia')
    counter = {}
    for value in values:
        if value in counter:
            counter[value] += 1
        else:
            counter[value] = 1
    
    md = max(counter.values())
    return [(v,f) for v,f in counter.items() if f >= md]

def amp(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular amplitude para uma lista vazia')
    h = max(values) - min(values)
    if decimals:
        return round(h, decimals)
    return h

def var(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular variância para uma lista vazia')
    xm = mean(values)
    s2 = sum([(x - xm) * (x - xm) for x in values]) / (len(values) - 1)

    if decimals:
        return round(s2, decimals)
    return s2

def std(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular desvio padrão para uma lista vazia')
    s2 = var(values)
    s = math.sqrt(s2)
    if decimals:
        return round(s, decimals)
    return s

def cvar(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular coeficiente de variação para uma lista vazia')
    coef = std(values) / mean(values) * 100
    if decimals:
        return round(coef, decimals)
    return coef

def quantile(values: list, percentual: float):
    if len(values) == 0:
        raise ValueError('Não se pode calcular quartis para uma lista vazia')
    list_sorted = sorted(values)
    idx = int(percentual * (len(values) - 1))
    return list_sorted[idx]

def iqr(values: list, decimals=None):
    if len(values) == 0:
        raise ValueError('Não se pode calcular amplitude interquartil para uma lista vazia')
    amp = quantile(values, .75) - quantile(values, .25)
    if decimals:
        return round(amp, decimals)
    return amp

def boxplot(values: list):
    plt.figure(figsize=(15,5))
    sns.boxplot(values, orient='h')
    plt.show()

def ic(values: list, alpha=.05, decimals=None):
    x = mean(values)
    s = std(values)
    n = len(values)  
    if n < 30:
        norm = st.t.ppf(q=1. - alpha / 2., df=n-1)
    else:
        norm = st.norm.ppf(1. - alpha / 2.)

    li = (x - (s * norm / math.sqrt(n)))
    ls = (x + (s * norm / math.sqrt(n)))

    if decimals:
        return (round(li, decimals), round(ls, decimals))
    return (li, ls)

def graus_liberdade(var_a, var_b, na, nb, decimals = None):
    # numerador
    term_va = (var_a / na) 
    term_vb = (var_b / nb)
    numV = term_va + term_vb
    numV *= numV
    # denominador
    denoV = ((1 / (na-1) * (term_va * term_va)) + (1 / (nb-1) * (term_vb * term_vb)))

    v = (numV / denoV) - 2
    if decimals:
        return round(v, decimals)
    return v

def ic_n_pareado(va: list, vb: list, alpha=.05, decimals=None):
    na = len(va)
    nb = len(vb)

    #1) diferença das médias
    xa = mean(va)
    xb = mean(vb)

    #2) diferença das médias
    dif = xa - xb

    #3) desvio padrão das amostras
    sa = std(va, decimals)
    sa *= sa # como as formúlas utilizam a variância, é mais prático otimizar
    sb = std(vb, decimals)
    sb *= sb # como as formúlas utilizam a variância, é mais prático otimizar

    #4) desvio padrão da diferença das médias
    s = math.sqrt((sa / na) + (sb / nb))

    #5) número de graus de liberdade
    v = graus_liberdade(sa , sb, na, nb, decimals)
    
    #6) IC da diferença das médias
    norm = st.t.ppf(q=(1. - alpha / 2.), df=v)

    li = (dif - (s * norm))
    ls = (dif + (s * norm))

    if decimals:
        return (round(li, decimals), round(ls, decimals))
    return (li, ls)

def t_zero(va: list, vb: list, alpha=.05, decimals=None):
    if len(va) == len(vb):
        return ic([a - b for a,b in zip(va,vb)], alpha, decimals)
    else:
        return ic_n_pareado(va, vb, alpha, decimals)

def sample_size(h, sigma, alpha=.05):
    # Calcula o valor z correspondente ao nível de confiança
    valor_z = abs(st.norm.ppf((1. - alpha / 2.)))
    # Calcula o tamanho da amostra
    return math.ceil((2 * valor_z * sigma / h) ** 2)

def shapiro_wilk(values: list, alpha=0.05):
    _, p = st.shapiro(values)
    if p > alpha:
        return True
    return False

def e_bernoulli(p):
    return p

def var_bernoulli(p):
    return p*(1-p)

def cv_bernoulli(p):
    return (math.sqrt(p*(1-p)))/p

def p_geometrica(p, i, decimals = None):
    r = p*pow((1-p), i-1)
    if(decimals):
        return round(r, decimals)
    return r

def e_geometrica(p, decimals = None):

    if (decimals):
        return round(1/p, decimals)
    return 1/p

def var_geometrica(p, decimals = None):

    r = (1-p)/p*p
    if(decimals):
        return round(r, decimals)
    return r

def cv_geometrica(p, decimals = None):

    r = (math.sqrt((1-p)/(p*p)))/(1/p)
    if (decimals):
        return round(r, decimals)
    return r

class G5RandomGenerator():
    
    def __init__(self, seed=0):
        self.__n0 = seed
        random.seed(self.__n0)
    
    # gera o próximo número com base no último número gerado
    def next(self):
        return random.random()
    
    # reinicia o gerador
    def reset(self):
        random.seed(self.__n0)

def va_exp(beta, gerador):
    U = gerador.next()
    return -beta * math.log(1 - U)

def valor_esperado(lamb, mi):
    p = lamb / mi
    return ((1 / mi) / (1-p)) * p




