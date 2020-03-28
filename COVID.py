
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim


from sklearn.linear_model import LinearRegression


url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

covid_data = pd.read_csv(url,error_bad_lines=False)

indice_base = covid_data[covid_data.keys()[0:4]]
indice_base['Date'] = covid_data.keys()[5]
indice_base['Cases'] = covid_data[covid_data.keys()[5]]

for i in range(6,len(covid_data.keys())):
            
    indice_apnd = covid_data[covid_data.keys()[0:4]]
    indice_apnd['Date'] = covid_data.keys()[i]
    indice_apnd['Cases'] = covid_data[covid_data.keys()[i]]
        
    indice_base = indice_base.append(indice_apnd)

br_covid = indice_base[ indice_base['Country/Region'] == 'Brazil']

teste = pd.Series(np.array(br_covid['Cases']),index=pd.date_range(br_covid['Date'].iloc[0],periods=len(br_covid['Date'])))


br_covid = br_covid.reset_index().set_index(np.array(pd.to_datetime(br_covid['Date']))).drop('index',axis=1).loc['2020-02-26':]

br_covid['Ln_Cases'] = np.log(br_covid['Cases'])



br_covid['Dys_1st_case'] = [i for i in range(0,len(br_covid))]

# Primeira tentativa de modelo é a de se fazer uma regressão linear simples com parametro de casos por dia convertido para 
# o logaritmo neperiano do númeri dos casos, isso nos permite adotar a forma funcional de y = a*e^b*x, ou seja, estamos adotando
# a premissa que a epidemia segue uma tendência exponencial. Aplicando o logaritmo neperianos nos dois lados da equação teremos:
# ln(y) = ln(a) + b*x

#cria array de variaveis (Dias desde o primeiro caso)


model_1 = LinearRegression(fit_intercept=True).fit(br_covid['Dys_1st_case'].values.reshape(-1, 1),
                                                     br_covid['Ln_Cases'].values.reshape(-1, 1))


predict_ln_cases = model_1.predict(br_covid['Dys_1st_case'].values.reshape(-1, 1))

br_covid['Predicted_Ln_Cases'] = predict_ln_cases
br_covid['Predicted_Cases'] = np.exp(predict_ln_cases)


plt.plot(pd.Series(np.array(br_covid['Cases']),index=pd.date_range(br_covid['Date'].iloc[0],periods=len(br_covid['Date']))))
plt.plot(pd.Series(np.array(br_covid['Predicted_Cases']),index=pd.date_range(br_covid['Date'].iloc[0],periods=len(br_covid['Date']))))
plt.show()

plt.scatter(x=np.array(br_covid['Dys_1st_case']), y= np.array(br_covid['Ln_Cases']))    
plt.plot(np.array(br_covid['Dys_1st_case']),np.array(br_covid['Predicted_Ln_Cases']),c= "red", marker='.', linestyle=':')
plt.show()

acu_cases = np.array(br_covid['Cases'])

inc_cases = [acu_cases[0]]

for i in range(1,len(acu_cases)):
    
    inc_cases.append( acu_cases[i]-acu_cases[i-1] )

plt.bar(pd.date_range(br_covid['Date'].iloc[1],periods=len(br_covid['Date'])),inc_cases)
plt.show()                      

### Aplicando um modelo de crescimento logistico ###

def logistic_func(t,a,b,c):
    
    return c/(1+a*np.exp(-b*t))

intervalos = (0, [100000.,3.,200000000])
par = np.random.exponential(size=3)

### Usando outros modelos para estimação dos casos ###

(a,b,c),cov = optim.curve_fit(logistic_func,
                              np.array(br_covid['Dys_1st_case'])+1,
                              np.array(br_covid['Cases']),
                              bounds=intervalos,
                              p0=par)

def logistic_func(t):
    
    return c/(1+a*np.exp(-b*t))

predict_logistic = [ logistic_func(t) for t in np.array(br_covid['Dys_1st_case'])]

plt.plot(np.array(br_covid['Dys_1st_case']),predict_logistic)
plt.plot(np.array(br_covid['Dys_1st_case']),np.array(br_covid['Cases']))
plt.show()

# Simulando o comportamento da curva durante o periodo de 90 dias após a infeção #

simul_logistic = [logistic_func(t) for t in range(1,91)]


plt.plot(pd.Series(simul_logistic, index=pd.date_range(br_covid['Date'].iloc[0],periods=90)))
plt.show()

print(a,b,c)


# In[44]:


print(par)


# In[91]:



### Tentando modelar uma curve logistica para os casos de infeção na china ###

ch_covid = indice_base[ indice_base['Country/Region'] == 'China']

ch_covid['Date'] = pd.to_datetime(ch_covid['Date'])

ch_covid = ch_covid.groupby(['Country/Region','Date'],as_index=False).sum().drop(['Lat','Long'],axis=1)



def m_logistic(t,a,b,c):
    
    return c/(1+a*np.exp(-b*t))

intervalos = (0, [100000.,3.,1000000000.])

p0 = np.random.exponential(size=3)

### Usando outros modelos para estimação dos casos ###

(a,b,c),cov = optim.curve_fit(m_logistic,
                              np.array(ch_covid.index)+1,
                              np.array(ch_covid['Cases']),
                              bounds=intervalos,
                              p0=p0)

def m_logistic(t):
    
    return c/(1+a*np.exp(-b*t))

predicted_china = [ m_logistic(t) for t in np.array(ch_covid.index)]

plt.plot(pd.Series(np.array(ch_covid['Cases']), index=np.array(ch_covid['Date'])))
plt.plot(pd.Series(predicted_china,  index=np.array(ch_covid['Date']) ))
plt.show()



# In[90]:


print(a,b,c)

