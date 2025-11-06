import numpy as np

def obter_dados_csv(nome_arquivo, lista_sinais = ['U_PV_1', 'I_PV_1'], fs = 500):
  '''
  nome_arquivo: string
  lista_sinais: nome dos sinais que se deseja obter
  saída 'sinais' é um vetor
  '''
  df = pd.read_csv(nome_arquivo, sep = ' ')
  t = np.arange(len(df[lista_sinais[0]]))/fs;
  sinais = []
  for sinal in lista_sinais:
    sinais.append(df[sinal])
  return t, sinais

def LPF(x, fc = 2, fs = 500, perc_init_value = 0.1):
    """
    Filtro passa-baixas (Low Pass Filter) de 1ª ordem
    x: sinal de entrada
    fc: frequência de corte (em Hz)
    fs: frequência de amostragem (em Hz)
    perc_init_value: porcentagem do sinal de entrada em que é realizada média para obter valor inicial
    saída: sinal filtrado  y
    """
    from scipy.signal import lfilter, lfilter_zi
    from numpy import exp, pi, mean
    alphaCo = 1 - exp(-2*pi*fc/fs)
    b = [alphaCo, 0]
    a = [1, -(1-alphaCo)]
    zi = lfilter_zi(b, a) * mean(x[0:int(len(x)*perc_init_value)])
    y,_ = lfilter(b, a, x, zi=zi)
    return y

def LPF2(x, fc = 2, fs = 500, perc_init_value = 0.1):
    """
    Filtro passa-baixas (Low Pass Filter) de 2ª ordem
    x: sinal de entrada
    fc: frequência de corte (em Hz)
    fs: frequência de amostragem (em Hz)
    perc_init_value: porcentagem do sinal de entrada em que é realizada média para obter valor inicial
    saída: sinal filtrado  y
    """
    return LPF(LPF(x, fc, fs, perc_init_value), fc, fs, perc_init_value)

def LPF3(x, fc = 2, fs = 500, perc_init_value = 0.1):
    """
    Filtro passa-baixas (Low Pass Filter) de 2ª ordem
    x: sinal de entrada
    fc: frequência de corte (em Hz)
    fs: frequência de amostragem (em Hz)
    perc_init_value: porcentagem do sinal de entrada em que é realizada média para obter valor inicial
    saída: sinal filtrado  y
    """
    return LPF2(LPF(x, fc, fs, perc_init_value), fc, fs, perc_init_value)

def HPF(x, fc = 2, fs = 500, perc_init_value = 0.1):
    """
    Filtro passa-altas (High Pass Filter) de 1ª ordem
    x: sinal de entrada
    fc: frequência de corte (em Hz)
    fs: frequência de amostragem (em Hz)
    perc_init_value: porcentagem do sinal de entrada em que é realizada média para obter valor inicial
    saída: sinal filtrado  y
    """
    return x - LPF(x, fc, fs, perc_init_value)

def HPF2(x, fc = 2, fs = 500, perc_init_value = 0.1):
    """
    Filtro passa-altas (High Pass Filter) de 1ª ordem
    x: sinal de entrada
    fc: frequência de corte (em Hz)
    fs: frequência de amostragem (em Hz)
    perc_init_value: porcentagem do sinal de entrada em que é realizada média para obter valor inicial
    saída: sinal filtrado  y
    """
    return x - LPF2(x, fc, fs, perc_init_value)


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interativo(t, y1, y2=None, y3=None, nome1='Sinal 1', nome2=None, nome3=None,
                    titulo=None, unidade_x='s', unidade_y='', template='plotly_white'):
    """
    Gera um gráfico interativo com até 3 curvas no mesmo eixo.

    Parâmetros:
    -----------
    t : array-like
        Eixo x (ex: tempo)
    y1, y2, y3 : array-like
        Sinais a serem plotados
    nome1, nome2, nome3 : str
        Nomes de cada curva (legenda)
    titulo : str
        Título do gráfico (opcional)
    unidade_x, unidade_y : str
        Unidades dos eixos
    template : str
        Tema visual do Plotly
    """

    fig = go.Figure()

    # Adiciona o primeiro sinal (obrigatório)
    fig.add_trace(go.Scatter(x=t, y=y1, mode='lines', name=nome1, line=dict(width=2)))

    # Segundo sinal (opcional)
    if y2 is not None:
        fig.add_trace(go.Scatter(x=t, y=y2, mode='lines', name=nome2 or 'Sinal 2', line=dict(width=2)))

    # Terceiro sinal (opcional)
    if y3 is not None:
        fig.add_trace(go.Scatter(x=t, y=y3, mode='lines', name=nome3 or 'Sinal 3', line=dict(width=2)))

    # Layout
    fig.update_layout(
        title=titulo or "Gráfico Interativo",
        xaxis_title=f"Tempo [{unidade_x}]",
        yaxis_title=f"Amplitude [{unidade_y}]" if unidade_y else "Amplitude",
        template=template,
        hovermode='x unified',
        legend=dict(title="Sinais", orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),

    fig.show()

def plotVI_interativo(t, v1, v2=None, i1=None, i2=None, nome_v1='Tensão 1', nome_v2='v2',
    nome_i1='i1', nome_i2='i2', titulo='Tensões e Correntes', unidade_x='s', unidade_v='V', unidade_i='A',
    template='plotly_white' ):
    """
    Gera um gráfico interativo com 2 tensões (eixo esquerdo) e 2 correntes (eixo direito).

    Parâmetros:
    -----------
    t : array-like
        Eixo x (ex: tempo)
    v1, v2 : array-like
        Tensões a serem plotadas (eixo esquerdo)
    i1, i2 : array-like
        Correntes a serem plotadas (eixo direito)
    nome_v1, nome_v2, nome_i1, nome_i2 : str
        Nomes de cada curva (legenda)
    titulo : str
        Título do gráfico (opcional)
    unidade_x, unidade_v, unidade_i : str
        Unidades dos eixos
    template : str
        Tema visual do Plotly
    """
    fig = go.Figure()

    # --- Tensões (Eixo esquerdo) ---
    fig.add_trace(go.Scatter(  x=t, y=v1, mode='lines', name=nome_v1, line=dict(width=2, color='royalblue'),  yaxis='y1' ))
    if v2 is not None:
        fig.add_trace(go.Scatter(  x=t, y=v2, mode='lines', name=nome_v2, line=dict(width=2, color='deepskyblue'),  yaxis='y1' ))

    # --- Correntes (Eixo direito) ---
    if i1 is not None:
        fig.add_trace(go.Scatter( x=t, y=i1, mode='lines', name=nome_i1, line=dict(width=2, color='firebrick'), yaxis='y2' ))
    if i2 is not None:
        fig.add_trace(go.Scatter( x=t, y=i2,mode='lines', name=nome_i2, line=dict(width=2, color='orange'),  yaxis='y2'  ))

    # --- Layout ---
    fig.update_layout( title=titulo, xaxis=dict(title=f"Tempo [{unidade_x}]"),
        yaxis=dict(title=f"Tensão [{unidade_v}]", side='left'),
        yaxis2=dict(title=f"Corrente [{unidade_i}]", overlaying='y', side='right'),
        template=template,  hovermode='x unified',
        legend=dict(  title="Sinais", orientation='h', yanchor='bottom', y=1.02,  xanchor='right', x=1 )
        )
    fig.show()


import matplotlib.pyplot as plt

def plot_ty1y2(t,y1,y2,figure=1, title = '',label1='vpv (V)', label2 = 'ipv (A)'):
  plt.figure(figure,figsize=(10,5))
  plt.title(title)
  plt.subplot(2,1,1)
  plt.plot(t,y1,label=(label1))
  plt.legend();  plt.grid();
  plt.subplot(2,1,2)
  plt.plot(t,y2,label = (label2))
  plt.legend();  plt.grid();

def plot_ty1(t,y1,figure=1, title = '',label1='vpv (V)'):
  plt.figure(figure,figsize=(10,5))
  plt.title(title)
  plt.plot(t,y1,label=(label1))
  plt.legend();  plt.grid();

def plotVI(v, i, figure = 1, title = '', legend = ('i (A)'),
           xlabel = 'tensão (V)', ylabel = 'corrente (A)'):
  plt.figure(figure);
  if type(v) is list:
      for q in range(len(v)):
          plt.plot(v[q],i[q])
  else:  plt.plot(v,i);
  plt.title(title);
  plt.grid();
  plt.xlabel(xlabel)
  plt.ylabel(ylabel);
  plt.legend(legend);


# Sistemas fotovoltaicos:

Params = {  'Voc_stc': 700.0,   #tensão de circuito aberto de 1 string
          'Isc_stc': 16.0,    #corrente de curto-circuito de 1 string
          'FFv':  0.83,        #fator de forma de tensão: Limites: 0.8 a 0.84
          'FFi': 0.93,        #Fator de forma de corrente: Limites: 0.9 a 0.95
          'Rd_Ω': 1.0,       #resistência total do diodo de bypass.  Não mudar.
          'G_W_m2': 1000.0,    #irradiância total. Limites: 0 a 1200. Default 1000
          'Temp_oC': 25.0,    #temperatura dos módulos fotovoltaicos. Limites: 10 a 70
          'gap_mm': 0.8,       #distância do arco. Limites: 0.8 a 2.5 (modelo atual)
          'shad_frac':0.1,  #fração do sistema com sombreamento (quando existente). Limites: 0 a 1
          'Gshad_frac':0.15, #fração entre irradiância da sombra / irradiância total. Limites: 0 a 0.5
          'shad_frac_str1': 0.05,     #parâmetros de sombreamento string 1
          'Gshad_frac_str1': 0.15,
          'shad_frac_str2': 0.0,    #parâmetros de sombreamento string 2
          'Gshad_frac_str2': 0.15,
          }

def PV_consts_IEC62891(Params: dict):
  G, T = Params['G_W_m2'], Params['Temp_oC']
  FFv, FFi = Params['FFv'], Params['FFi']
  Cg, Cv, Cr = 2.514e-3, 8.593e-2, 1.088e-4
  vL2H, α, β, Gstc, Tstc = 0.95, 0.045e-2, -0.275e-2, 1000.0, 25.0
  Isc = Params['Isc_stc']*G/Gstc*(1 + α*(T-Tstc));
  Voc = Params['Voc_stc']*(1+β*(T-Tstc))*(np.log(G/Cg+1)*Cv-Cr*G)
  Io = Params['Isc_stc']*(1-FFi)**(1/(1-FFv))*G/Gstc
  Caq = (FFv-1)/np.log(1-FFi)
  return Isc, Voc, Io, Caq

def calc_ipv_IEC62891_A(vpv:float, Params: dict) -> float: #Calcula ipv = f(vpv)
  Isc, Voc, Io, Caq = PV_consts_IEC62891(Params)
  return ((Isc + Io - Io*np.exp(vpv/(Voc*Caq)))*(vpv>0) + (Isc-vpv/Params['Rd_Ω'])*(vpv<=0))

def calc_vpv_IEC62891_V(ipv:float, Params: dict) -> float: #Calcula vpv = g(ipv)
  Isc, Voc, Io, Caq = PV_consts_IEC62891(Params)
  ipvd, ipvpv = ((ipv-Isc)*(ipv>Isc)) ,  (ipv*(ipv<=Isc)+Isc*(ipv>Isc))
  vpv = Voc*Caq*np.log((Isc+Io-ipvpv)/Io)  -  ipvd*Params['Rd_Ω']
  return vpv

def calc_curva_VI_1string(Params:dict):
  vpv = np.arange(0,Params['Voc_stc'],1)
  return vpv, calc_ipv_IEC62891_A(vpv, Params)

def calc_curva_VI_2strings(Params: dict):
  Vpv,Ipv = calc_curva_VI_sem_arco_1string(Params)
  return Vpv, 2*Ipv


def calc_vpv_part_shad_serie(ipv:float, Params:dict):
  PA, PB = Params.copy(), Params.copy();
  PA['Voc_stc'] = PA['Voc_stc']*(1-PA['shad_frac'])   #sem sombra
  PA['Rd_Ω'] = PA['Rd_Ω']*(1-PA['shad_frac'])   #sem sombra
  PB['Voc_stc'] = PB['Voc_stc']*PB['shad_frac']       #com sombra
  PA['Rd_Ω'] = PB['Rd_Ω']*(1-PB['shad_frac'])   #sem sombra
  PB['G_W_m2'] = PB['G_W_m2']*PB['Gshad_frac']        #irradiância com sombra
  return calc_vpv_IEC62891_V(ipv, PA) + calc_vpv_IEC62891_V(ipv, PB)

def calc_curva_VI_part_shad_serie(Params:dict):
  Iscmax,_,_,_ = PV_consts_IEC62891(Params)
  ipv = np.arange(0, Iscmax+1, 0.01)
  vpv = calc_vpv_part_shad_serie(ipv, Params)
  return vpv, ipv

def calc_ipv_part_shad_serie(vpv:float, Params:dict):
  '''
  Utiliza o método de Newton com ganho reduzindo após a 5ª iteração
  '''
  PA = Params.copy()
  PA['Voc_stc'] = PA['Voc_stc']*(1-PA['shad_frac'])
  ipv = calc_ipv_IEC62891_A(vpv*(1-PA['shad_frac']), PA)  #valor inicial
  for q in range(0,20):  #iterações
    step = 1e-6/(1+(q-5)*(q>5))   #ganho reduzindo após q>5
    dfdx = (calc_vpv_part_shad_serie(ipv,Params) -
            calc_vpv_part_shad_serie(ipv-1e-6,Params))/step
    Δipv = - np.clip( (calc_vpv_part_shad_serie(ipv,Params)-vpv)/dfdx, -2, 2)
    ipv = ipv  + Δipv
  return ipv

def calc_curva_VI_part_shad_2strings(Params: dict):
  '''
  shad_frac_str1, Gshad_frac_str1: parâmetros de sombreamento da string 1
  shad_frac_str2, Gshad_frac_str2: parâmetros de sombreamento da string 2
  '''
  Iscmax,Vocmax,_,_ = PV_consts_IEC62891(Params)
  vpv = np.arange(0,Params['Voc_stc'],1)
  Ps1, Ps2 = Params.copy(), Params.copy()
  Ps1['shad_frac'], Ps1['Gshad_frac'] = Ps1['shad_frac_str1'], Ps1['Gshad_frac_str1']
  Ps2['shad_frac'], Ps2['Gshad_frac'] = Ps2['shad_frac_str2'], Ps2['Gshad_frac_str2']
  ipv =  calc_ipv_part_shad_serie(vpv, Ps1) + calc_ipv_part_shad_serie(vpv, Ps2)
  return vpv, ipv


# Arco elétrico:
def calc_Varc_IEC63027(iarc_A: float, gap_mm: float):
    '''
    Calcula a tensão do arco em função da corrente.
    Modelo válido de 3 A a 16 A
    '''
    z=1.959964 #Para intervalo de 95%
    i, g = iarc_A, gap_mm
    mu = 41.51100 + 0.91995*i + 3.05736*g - 11.58432*np.log(i) + 6.47092*np.log(g) - 0.18720*g*i
    s = np.exp(1.343885 - 0.113478*i -0.300667*g + 1.181893*np.log(g))
    li = mu - z*s
    ls = mu + z*s
    return mu, li, ls
