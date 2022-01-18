import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit



##############################################################
#función que simula la evolucón temporal del sistema
##############################################################
def probability_density(N,s,l,ensembles,time_steps,histogram_bins, view_times): # s: guardado de producción, l: Guardado de producción
    #Función que retorna la evolución temporal de los histogramas, normalizados por integral y suma de los valores de las barras del histograma, junto con los pasos de tiempo, los másximos y mínimos y la riqueza total
    #El esquema de computo consiste en aplicar la evolución temporal para todos los ensambles de forma simultanea y tomar como muestra cada view_times datos de tiempo


    wealth = np.ones((ensembles,N)) # Definimos la riqueza inicial de 1 para todos los agentes
    data_evol_view = np.zeros((int(time_steps/view_times),ensembles,N)) #Inicilización de la riqueza de los N agentes en todos los ensambles para cada paso de tiempo. Posteriormente se promediará sobre los ensambles para calcular la distribución
    hist_time_data = np.zeros((int(time_steps/view_times))) #Inicialización del array que guarda los valores de timepo de los pasos de tiempo que se consideraron (cada view_times pasos)
    total_wealth = np.zeros((int(time_steps/view_times))) # Inicialización del vector que almacena la riqueza en cada paso de timepo observado
    total_wealth[0] = wealth.sum()/ensembles #Añadimos la riqueza total inicial para el paso de tiempo 0 
    data_evol_view[0] = wealth #Añadimos la riqueza de todos los agentes y ensambles 

    aux_time_idx = 1 #Indice auxiliar para los pasos de tiempo observados
    rd.seed(1234) #Definimos una semilla para tener consistencia entre ejecuciones de la simulación
    
    # Para cada paso de tiempo y todos los ensambles evolucionamos la riqueza de cada uno de agentes segun la regla de intercambio.
    # Se toman un total de view_times de muestras en el tiempo uniformemente distriuidas
    for time in range(1,time_steps):    #Iteramos sobre cada paso de tiempo
        for idx_ens, ensemble in enumerate(wealth):     #Para cada paso de tiempo iteramos sobre todos los ensambles.
        
            i = rd.randint(0,N-1)   #Escogemos el agente i(j) para el interacambio con el j (i)
            j = rd.randint(0,N-1)
            if i == j: #Si se escoge el mismo agente se pasa al siguiente paso
                continue
            
            eps = rd.random()
            w_i = wealth[idx_ens,i] # Encontramos la riqueza del agente i y j en el ensamble que estamos 
            w_j = wealth[idx_ens,j]  

            delta_i = ((1-l)/(l-s))*l*(eps*(w_i+w_j)-w_i) + ((1-l)/(l-s))*s*w_i  #Calculamos los delta i y j segun la regla de intercambio
            delta_j = -((1-l)/(l-s))*l*(eps*(w_i+w_j)-w_i) + ((1-l)/(l-s))*s*w_j 
            
            wealth[idx_ens,i] += delta_i #Sumamos los cambios de riqueza a cada agente
            wealth[idx_ens,j] += delta_j
        
        if time % view_times == 0:   #Cada view_times pasos de tiempo guardamos los datos de la simulación

            hist_time_data[aux_time_idx] = time     # Guardamos el paso de tiempo
            data_evol_view[aux_time_idx] = wealth   # Guardamos los ensambles de riqueza para ese paso de tiempo
            total_wealth[aux_time_idx] = wealth.sum()/ensembles     #Guardamos a riqueza total
            aux_time_idx += 1   
    

    
    # Dados las muestras en el tiempo cosntruimos los histogramas de cada paso de tiempo promediando sobre los ensambles
    hist_time_evol = np.zeros((int(time_steps/view_times),histogram_bins))  #Inicialización de donde se guardan los histogramas para cada paso de tiempo, normalizados por integración y por suma de histogramas
    hist_time_evol_norm = np.zeros((int(time_steps/view_times),histogram_bins)) 
    
    max_hist = data_evol_view.max() #Encontramos los máximos y mínimos para hacer el histograma comun entre  los ensambles y tiempos
    min_hist = data_evol_view.min()

    #Encontramos los histogramas promediados, con normalización de la integral para obtener la densidad de probabilidad continua  y la normaliada a 1 al sumar los valores del histograma
    for idx, time_view in enumerate(data_evol_view):
        sum = np.zeros(histogram_bins)
        sum_norm = np.zeros(histogram_bins)
        for idx_ens, ensemble in enumerate(time_view):
            hist = np.histogram(ensemble,histogram_bins,(min_hist,max_hist), density = True)
            sum += hist[0]

            hist_norm = np.histogram(ensemble,histogram_bins,(min_hist,max_hist))
            sum_norm += hist_norm[0]/hist_norm[0].sum()

        sum /= ensembles
        sum_norm /= ensembles

        # hist_time_evol[idx] = sum/((max_hist-min_hist)/histogram_bins)
        hist_time_evol[idx] = sum
        hist_time_evol_norm[idx] = sum_norm

    
    return hist_time_evol,hist_time_evol_norm, total_wealth ,hist_time_data, min_hist, max_hist



##############################################################
#función que simula el caso de dos tipos de agentes
##############################################################

def probability_density_workers(N,alpha,producers,s,l,ensembles,time_steps,histogram_bins, view_times): # s: guardado de producción, l: Guardado de producción
    
    #Repetimos el procedimiento de un solo agente pero duplicando las cantidades para tomar dos agentes, simplemente se modificam las reglas de intercambio


    wealth_producers = np.ones((ensembles,int(producers*N)))
    wealth_workers = np.ones((ensembles,int(producers*N)))
    
    data_evol_view_producers = np.zeros((int(time_steps/view_times),ensembles,int(producers*N))) 
    data_evol_view_workers = np.zeros((int(time_steps/view_times),ensembles,int((producers)*N)))
    
    hist_time_data = np.zeros((int(time_steps/view_times)))
    
    total_wealth_producers = np.zeros((int(time_steps/view_times)))
    total_wealth_workers = np.zeros((int(time_steps/view_times)))
    
    total_wealth_producers[0] = wealth_producers.sum()/ensembles
    total_wealth_workers[0] = wealth_workers.sum()/ensembles
    
    data_evol_view_producers[0] = wealth_producers 
    data_evol_view_workers[0] = wealth_workers 

    Nl_Np = int((1-producers)/producers)
    aux_time_idx = 1
    rd.seed(1234)
    
    # Para cada paso de tiempo y todos los ensambles evolucionamos la riqueza de cada uno de agentes segun la regla de intercambio.
    # Se toman un total de view_times de muestras en el tiempo uniformemente distriuidas
    for time in range(1,time_steps):
        for idx_ens, ensemble in enumerate(wealth_producers):
            i = rd.randint(0,int(N*producers)-1)
            j = rd.randint(0,int(N*producers)-1)
            if i == j:
                continue
            
            eps = rd.random()
            w_i = wealth_producers[idx_ens,i]
            w_j = wealth_producers[idx_ens,j]

            delta_i = ((1-l)/(l-s))*l*(eps*(w_i+w_j)-w_i) + ((1-l)/(l-s))*s*w_i 
            delta_j = -((1-l)/(l-s))*l*(eps*(w_i+w_j)-w_i) + ((1-l)/(l-s))*s*w_j
            
            wealth_producers[idx_ens,i] += alpha*delta_i
            wealth_producers[idx_ens,j] += alpha*delta_j

            wealth_workers[idx_ens,i] += 1/(2*Nl_Np)*(1-alpha)*(delta_i+delta_j)
            wealth_workers[idx_ens,j] += 1/(2*Nl_Np)*(1-alpha)*(delta_i+delta_j)

        if time % view_times == 0:

            hist_time_data[aux_time_idx] = time
            data_evol_view_producers[aux_time_idx] = wealth_producers
            data_evol_view_workers[aux_time_idx] = wealth_workers

            total_wealth_producers[aux_time_idx] = wealth_producers.sum()/ensembles
            total_wealth_workers[aux_time_idx] = wealth_workers.sum()/ensembles
            
            aux_time_idx += 1
    

    
    # Dados las muestras en el tiempo cosntruimos los histogramas de cada paso de tiempo promediando sobre los ensambles
    hist_time_evol_producers = np.zeros((int(time_steps/view_times),histogram_bins))
    hist_time_evol_workers = np.zeros((int(time_steps/view_times),histogram_bins))
    hist_time_evol_norm_producers = np.zeros((int(time_steps/view_times),histogram_bins))
    hist_time_evol_norm_workers = np.zeros((int(time_steps/view_times),histogram_bins))
    
    max_hist_producers = data_evol_view_producers.max()
    max_hist_workers = data_evol_view_workers.max()
    min_hist_producers = data_evol_view_producers.min()
    min_hist_workers = data_evol_view_workers.min()


    for idx, (time_view_producers, time_view_workers) in enumerate(zip(data_evol_view_producers,data_evol_view_workers)):
        sum_producers = np.zeros(histogram_bins)
        sum_workers = np.zeros(histogram_bins)
        sum_norm_producers = np.zeros(histogram_bins)
        sum_norm_workers = np.zeros(histogram_bins)

        for idx_ens, (ensemble_producers, ensemble_workers) in enumerate(zip(time_view_producers,time_view_workers)):
            hist_producers = np.histogram(ensemble_producers,histogram_bins,(min_hist_producers,max_hist_producers), density = True)
            hist_workers = np.histogram(ensemble_workers,histogram_bins,(min_hist_workers,max_hist_workers), density = True)
            sum_producers += hist_producers[0]
            sum_workers += hist_workers[0]

            hist_norm_producers = np.histogram(ensemble_producers,histogram_bins,(min_hist_producers,max_hist_producers))
            hist_norm_workers = np.histogram(ensemble_workers,histogram_bins,(min_hist_workers,max_hist_workers))
            sum_norm_producers += hist_norm_producers[0]/hist_norm_producers[0].sum()
            sum_norm_workers += hist_norm_workers[0]/hist_norm_workers[0].sum()


        sum_producers /= ensembles
        sum_workers /= ensembles
        sum_norm_producers /= ensembles
        sum_norm_workers /= ensembles

        # hist_time_evol[idx] = sum/((max_hist-min_hist)/histogram_bins)
        hist_time_evol_producers[idx] = sum_producers
        hist_time_evol_workers[idx] = sum_workers
        hist_time_evol_norm_producers[idx] = sum_norm_producers
        hist_time_evol_norm_workers[idx] = sum_norm_workers

    
    return [hist_time_evol_producers, hist_time_evol_workers, hist_time_evol_norm_producers, hist_time_evol_norm_workers, total_wealth_producers, total_wealth_workers, hist_time_data, min_hist_producers, min_hist_workers, max_hist_producers, max_hist_workers]




##############################################################
#función de distribución de probabilidad Gamma ajustada
##############################################################

def gamma_dist(x,a,b):
    return 1/(a*gamma(b))*(x/a)**(b-1)*np.exp(-x/a)

##############################################################
#función que encuentra elajuste de la distribución Gamma a los
#datos
##############################################################

def ajuste_gamma(hist,time,a0,b0,s,l, histogram_bins, min, max, initial_time):
    # Se obtiene el auste para los distintos pasos de tiempo
    width = (max-min)/histogram_bins
    hist_x = np.linspace(min+width/2,max-width/2,histogram_bins)

    a_vec = []
    b_vec = []
    time_ab = []

    for h, t  in zip(hist,time):
        if t >=initial_time:
            param = [a0,b0]
            popt, pcov = curve_fit(gamma_dist,hist_x,h,param)

            a0 = popt[0]
            b0 = popt[1]

            a_vec.append(popt[0])
            b_vec.append(popt[1])
            time_ab.append(t)


           
    return np.array(a_vec), np.array(b_vec), np.array(time_ab)

##############################################################
#función que calcula el momento r-esimo de una distribución
##############################################################

def moment_r(r,hist,min,max,histogram_bins):
    #Calcula el momento r dado un histograma cuya suma está normalizada a 1
    
    width = (max-min)/histogram_bins
    w = np.linspace(min+width/2,max-width/2,histogram_bins)
    momento = 0

    for idx, p in enumerate(hist):
        momento += (w[idx]**r)*p
    
    return momento

##############################################################
#función que calcula la evolución temporal del momento r-esimo
##############################################################

def moment_time(r,hist_time,time,min,max,histogram_bins):
    aux = []
    for idx, i in enumerate(time):
        moment = moment_r(r,hist_time[idx],min,max,histogram_bins)
        aux.append(moment)
    
    return np.array(aux)

##############################################################
#Función que calcula el momento teórico segun la teoría
##############################################################

def moment_r_teo(r,s,l,time):
    m2 = (l**2*(l+2-3*s))/(3*(l-s)*(l+s-l*s)-2*l**2*(1-l))
    g = ((1-l)/(l-s))*s
    tau = 2*time/N

    if r == 1:
        return np.exp(g*tau)
    
    if r == 2:
        return m2*np.exp(2*g*tau)

##############################################################
#función que calcula los momentos teoricos de los productores
#en el caso de dos tipos de agentes
##############################################################

def wealth_teo_workers(s,l,alpha,N_p,time):
    g_e = ((1-l)/(l-s))*alpha*s
    tau = 2*time/N_p

    return np.exp(g_e*tau)
    
##############################################################
#función que calcla los valores teoricos de a y b de la función
#Gamma dado en terminos de lambda y s
##############################################################

def ab_teo(s,l,time):
    
    m = (l**2*(l+2-3*s))/(3*(l-s)*(l+s-l*s)-2*l**2*(1-l))
    g = ((1-l)/(l-s))*s
    b = 1/(m-1)
    tau = 2*time/N
    a = (m-1)*np.exp(g*tau)
    return a, b

##############################################################
#función que calcula el índice de Gini para una distribución
##############################################################

def gini_index(h_norm,min,max,histogram_bins):
    X = []
    Y = []

    width = (max-min)/histogram_bins

    for idx,i in enumerate(h_norm):
        aux = moment_r(0,h_norm[:idx],min,max,histogram_bins)
        X.append(aux)
    
    M1 = moment_r(1,h_norm,min,max,histogram_bins)
    for idx,i in enumerate(h_norm):
        aux = moment_r(1,h_norm[:idx],min,max,histogram_bins)
        Y.append(aux/M1)
    
    dxdw = np.gradient(X)
    
    gini = 0 
    for idx, i in enumerate(X):
        gini += 2*(X[idx]-Y[idx])*dxdw[idx]

    return gini





##############################################################
#Simulación 1: simulación de un único agente con los parametros 
# mostrados a continuación
##############################################################


N =1000 #Numero de agentes N
# ensembles = 5*10**1 #Numero de ensambles promediados
ensembles = 5*10**3 #Numero de ensambles promediados
time_steps = 20100 #Pasos de tiempo considerados
histogram_bins = 10000 #Numero de divisiones para los histogramas que general la densidad de probabilidad
view_times = 100
init_time_ab = 10000


l = [0.5,0.5,0.8,0.7,0.8,0.8,0.8]   #Valores de lambda
s = [0.05,0.1,0.1,0.2,0,0.2,0.3]  #Valores de s
a0 = [1,3,0.1,0.5,0.1,2,2]        #a inicial para los ajustes 
b0 = [3,3,12,8,8,8,8]             #b inicial para los ajustes
hist = []                       #Vector que almacena en cada entrada los histogramas de cada para de parametros lambda y s. Es el histograma "normalizado" con density = True en np.histogram
hist_norm = []                  #Vector con los histogramas con la suma normalizada a 1
total_wealth = []               #Vector con la evolución temporal de la riqueza de cada l y s en cada entrada
time = []                       #Vector que guarda el paso de timepo de los instantes de tiempo que se estan guardando
min = []                        #Vector que guarda la riqueza mínima en cada entrada para cada l y s escogidos
max = []                        #Vector que guarda la riqueza maxima en cada entrada para cada l y s escogidos
a_ajuste = []                   #Vector que almacena en cada entrada la evolución en el tiempo del a ajustado para cada l y s
b_ajuste = []                   #Vector que almacena en cada entrada la evolución en el tiempo del b ajustado para cada l y s
time_ab = []                    #Valores de tiempo para los cuales se estan considerando los valores a y b

for idx, i in enumerate(l):
    result = probability_density(N, s[idx], l[idx], ensembles,time_steps,histogram_bins, view_times)
    
    hist.append(result[0])
    hist_norm.append(result[1])
    total_wealth.append(result[2])
    time.append(result[3])
    min.append(result[4])
    max.append(result[5])


    result = ajuste_gamma(hist[idx],time[idx],a0[idx],b0[idx],s[idx],l[idx], histogram_bins, min[idx], max[idx], init_time_ab)
    
    a_ajuste.append(result[0])
    b_ajuste.append(result[1])
    time_ab.append(result[2])

    print(idx)




##############################################################
#Grafica de algunos de las distribuciones de riqueza y para 
# distintos pasos de tiempo
##############################################################

time_idxs = range(15000,20100,1000)
conf_idxs = [0,1,2,3]

for idx_c in conf_idxs:

    width = (max[idx_c]-min[idx_c])/histogram_bins
    hist_x = np.linspace(min[idx_c]+width/2,max[idx_c]-width/2,histogram_bins)
    hist_x_ajuste = np.linspace(min[idx_c]+width/2,max[idx_c]-width/2,1000)
    for idx_t in time_idxs:
        plt.plot(hist_x,hist[idx_c][int(idx_t/100)],label = "Paso de tiempo: {}".format(idx_t),markersize = 6,marker = ".", markevery = 100, linestyle = "None" )
        if idx_t == 20000:
            plt.plot(hist_x_ajuste, gamma_dist(hist_x_ajuste, a_ajuste[idx_c][int((idx_t-init_time_ab)/100)],b_ajuste[idx_c][int((idx_t-init_time_ab)/100)]  ),color = "#637074", label = "Ajuste Dist. Gamma",zorder =-1)
        else:
            plt.plot(hist_x_ajuste, gamma_dist(hist_x_ajuste, a_ajuste[idx_c][int((idx_t-init_time_ab)/100)],b_ajuste[idx_c][int((idx_t-init_time_ab)/100)] ),color = "#637074",zorder = -1)

    
    plt.xlabel("Nivel de riqueza individual", fontsize = 13)
    plt.ylabel("Densidad de Probabilidad", fontsize = 13)
    plt.xticks(fontsize =12)
    plt.yticks(fontsize =12)
    plt.legend(fontsize = 12)
    plt.title("$\lambda$ = {}, $s$ = {}".format(l[idx_c],s[idx_c]), fontsize = 12)
    plt.savefig("distribucion_evolucion_temporal_l{}s{}.pdf".format(l[idx_c],s[idx_c]), bbox_inches='tight' )
    plt.show()




##############################################################
#Grafica de la evolución temporal de los parametros a y b
# teoricos y simulados en la primera simulación
##############################################################

for idx, i in enumerate(time_ab):

    plt.plot(time_ab[idx],a_ajuste[idx],label = "$\lambda$ = {}, $s$={}".format(l[idx],s[idx]), markevery = 2,markersize = 8, marker = ".", linestyle = "None")
    a_teo, b_teo = ab_teo(s[idx], l[idx], time_ab[idx])
    plt.plot(time_ab[idx],a_teo, color = "#637074",zorder = -1)
    plt.yscale("log")


plt.legend(ncol=3,loc=2, fontsize = 9)
plt.ylabel("Parametro de escala: $a$", fontsize = 13)
plt.xlabel("Pasos de tiempo", fontsize = 13)
plt.xticks(fontsize =12)
plt.yticks(fontsize =12)
plt.ylim([5*10**-2,800])
plt.savefig("a_time.pdf", bbox_inches='tight' )
plt.show()


for idx, I in enumerate(time_ab):

    plt.plot(time_ab[idx],b_ajuste[idx], label = "$\lambda$ = {}, $s$={}".format(l[idx],s[idx]), markevery = 2,markersize = 8, marker = ".", linestyle = "None")
    a_teo, b_teo = ab_teo(s[idx], l[idx], time_ab[idx])
    plt.plot([time_ab[idx][0],time_ab[idx][-1]],[b_teo,b_teo],color = "#637074",zorder = -1)

plt.ylabel("Parametro de forma: $b$", fontsize = 13)
plt.xlabel("Pasos de tiempo", fontsize = 13)
plt.xticks(fontsize =12)
plt.yticks(fontsize =12)
plt.legend(ncol=3,loc=2, fontsize = 9)
plt.ylim([2,17.8])
plt.savefig("b_time.pdf", bbox_inches='tight' )
plt.show()




##############################################################
#Grafica de la evolución temporal del primer y segundo momento
# de la primera simulación
##############################################################


for idx, i in enumerate(l):
    r = 1 #Calculamos el momento 1
    moment_time_exp = moment_time(r,hist_norm[idx],time[0],min[idx],max[idx],histogram_bins)
    moment_time_teo = moment_r_teo(r,s[idx],l[idx],time[0])
    plt.plot(time[0],moment_time_exp,label="l = {}, s = {}".format(l[idx],s[idx]),markevery = 5,marker =".", markersize = 8, linestyle = "None")
    if idx == 6:
        plt.plot(time[0],moment_time_teo, color = "#637074",zorder = -1,label = "$M_1 = \exp(g\\tau)$")
    else:
        plt.plot(time[0],moment_time_teo, color = "#637074",zorder = -1)

plt.ylabel("Primer Momento $M_1(t)$", fontsize = 13)
plt.xlabel("Pasos de tiempo", fontsize = 13)
plt.xticks(range(0,20100,5000),fontsize =12)
plt.yticks(fontsize =12)
plt.legend()
plt.yscale("log")  
plt.savefig("M1_time.pdf", bbox_inches='tight' )
plt.show()


for idx, i in enumerate(l):
    r = 2 #Calculamos el momento 2
    moment_time_exp = moment_time(r,hist_norm[idx],time[0],min[idx],max[idx],histogram_bins)
    moment_time_teo = moment_r_teo(r,s[idx],l[idx],time[0])
    plt.plot(time[0],moment_time_exp,label="l = {}, s = {}".format(l[idx],s[idx]),markevery = 5,marker =".", markersize = 8, linestyle = "None")
    if idx == 6:
        plt.plot(time[0],moment_time_teo, color = "#637074",zorder = -1, label = "$M_2 = m_2\exp(2g\\tau)$")
    else:
        plt.plot(time[0],moment_time_teo, color = "#637074",zorder = -1)

plt.ylabel("Primer Momento $M_2(t)$", fontsize = 13)
plt.xlabel("Pasos de tiempo", fontsize = 13)
plt.legend()
plt.xticks(range(0,20100,5000),fontsize =12)
plt.yticks(fontsize =12)
plt.yscale("log")
plt.savefig("M2_time.pdf", bbox_inches='tight' )
plt.show()



##############################################################
#Grafica de las distribuciones estacionarios. Tanto la 
# evolución temporal de una da las configuraciones lambda =0.7
# s = 0.2 así como el caso ya estabilizado para distintas 
# configuraciones en un paso de tiempo ya estable (t = 10000)
##############################################################
# for idx, i in enumerate(l):
z = 0

for idx in [3]:
    for f in [50,100,150,200]:
        col = ["r","g","b","p","k","brown"]
        for h,h_norm in zip(hist[idx][f:f+1],hist_norm[idx][f:f+1]):
            width = (max[idx]-min[idx])/histogram_bins
            promedio = moment_r(1,h_norm,min[idx],max[idx],histogram_bins)
            
            x = np.linspace(min[idx]+width/2,max[idx]-width/2,histogram_bins)/promedio
            y = h*promedio
            plt.plot(x,y,marker='.',markersize = 10,markevery =20,linestyle = "None", label = "Paso de tiempo: {}".format(f*100),zorder = -z)
            z+=1
            # integral = np.trapz(y,x)
            # print(integral)

plt.xlabel("Nivel de riqueza $w/\\langle w \\rangle$", fontsize = 13)
plt.ylabel("Densidad de Probabilidad ", fontsize = 13)       

plt.xlim([0,5])
plt.title("Distribución Cuasi-estacionaria $\lambda$ = {}, $s$ = {}".format(l[3],s[3]), fontsize = 12)
plt.legend(fontsize = 12)
plt.xticks(fontsize =12)
plt.yticks(fontsize =12)
plt.savefig("semi_estacionario_time.pdf", bbox_inches='tight' )
plt.show()


for idx in [4,2,5,6]:
    f = 100
    col = ["r","g","b","p","k","brown"]
    for h,h_norm in zip(hist[idx][f:f+1],hist_norm[idx][f:f+1]):
        width = (max[idx]-min[idx])/histogram_bins
        promedio = moment_r(1,h_norm,min[idx],max[idx],histogram_bins)
        
        x = np.linspace(min[idx]+width/2,max[idx]-width/2,histogram_bins)/promedio
        y = h*promedio
        plt.plot(x,y,marker='.',markevery = 10,linestyle = "None", label = "$\lambda$ = {}, $s$= {}".format(l[idx],s[idx]))
        x_ajuste = np.linspace(0,100,3000)
        y_ajuste = gamma_dist(x_ajuste/promedio,a_ajuste[idx][0]/promedio,b_ajuste[idx][0])#Distribución Gamma cuasi-estatica
        if idx == 6:
            plt.plot(x_ajuste/(promedio),y_ajuste,color ="#212227",label = "Ajuste Distribución Gamma",zorder = 10)
        else:
            plt.plot(x_ajuste/(promedio),y_ajuste,color ="#212227",zorder = 10)

plt.xlabel("Nivel de riqueza  $w/\\langle w \\rangle$", fontsize = 13)
plt.ylabel("Densidad de Probabilidad", fontsize = 13)       
plt.title("Distribución Cuasi-estacionaria", fontsize = 12 )
plt.xlim([0,5])
plt.legend(fontsize = 12)
plt.xticks(fontsize =12)
plt.yticks(fontsize =12)
plt.savefig("semi_estacionario_configuraciones.pdf", bbox_inches='tight' )
plt.show()




##############################################################
#Calculo del indice de gini para la distribución estatica de 
# lambda = 0.7 y s = 2
##############################################################

gini_vec = []

for idx in [3]:
    gini_time =[]
    for idxt,t in enumerate(time[0]):
        if idxt%5 == 0:
            res = gini_index(hist_norm[idx][idxt],min[idx],max[idx],histogram_bins)
            gini_time.append(res)
    gini_vec.append(gini_time)

##############################################################
#Grafica del indice de gini cuasi-estatico en el tiempo
##############################################################

plt.plot(np.linspace(0,20000,41),gini_vec[0], markersize = 10,marker = ".", linestyle = "None")
plt.xlabel("Pasos de tiempo", fontsize=13)
plt.ylabel("Indice de Gini", fontsize=13)
plt.title("$\lambda$ = {}, $s$={}".format(0.7,0.2),fontsize = 12)
plt.xticks(range(0,20100,5000),fontsize = 12)
plt.xticks(fontsize = 12)
plt.savefig("gini_estatico.pdf",bbox_inches="tight")
plt.show()




##############################################################
#simulación 2: Simulación con mutiples lambda y s para graficar 
# la dependencia del indice de gini con lambda y s
##############################################################

N =1000 #Numero de agentes N
ensembles = 10**2#Numero de ensambles promediados
time_steps = 20100 #Pasos de tiempo considerados
histogram_bins = 1000 #Numero de divisiones para los histogramas que general la densidad de probabilidad


view_times = 100
init_time_ab = 10000


l = [*[0.2]*10,*[0.35]*10,*[0.5]*10]   #Valores de lambda
s = [*np.linspace(0,0.1,10),*np.linspace(0,0.2,10),*np.linspace(0,0.3,10)]  #Valores de s
hist = []                       #Vector que almacena en cada entrada los histogramas de cada para de parametros lambda y s. Es el histograma "normalizado" con density = True en np.histogram
hist_norm = []                  #Vector con los histogramas con la suma normalizada a 1
total_wealth = []               #Vector con la evolución temporal de la riqueza de cada l y s en cada entrada
time = []                       #Vector que guarda el paso de timepo de los instantes de tiempo que se estan guardando
min = []                        #Vector que guarda la riqueza mínima en cada entrada para cada l y s escogidos
max = []                        #Vector que guarda la riqueza maxima en cada entrada para cada l y s escogidos
a_ajuste = []                   #Vector que almacena en cada entrada la evolución en el tiempo del a ajustado para cada l y s
b_ajuste = []                   #Vector que almacena en cada entrada la evolución en el tiempo del b ajustado para cada l y s
time_ab = []                    #Valores de tiempo para los cuales se estan considerando los valores a y b

for idx, i in enumerate(l):
    result = probability_density(N, s[idx], l[idx], ensembles,time_steps,histogram_bins, view_times)
    
    hist.append(result[0])
    hist_norm.append(result[1])
    total_wealth.append(result[2])
    time.append(result[3])
    min.append(result[4])
    max.append(result[5])




##############################################################
#calculo del indice de Gini para la simulación 2
##############################################################
y = []
for idx,h in enumerate(hist_norm):
    aux = gini_index(h[-1],min[0],max[0],histogram_bins)
    y.append(aux)

##############################################################
#Grafica de los indices de gini para la simulación 2
##############################################################

gini_l02 = y[0:10]
s_l02 = s[0:10]
gini_l035 = y[10:20]
s_l035 = s[10:20]
gini_l05 = y[20:]
s_l05 = s[20:]

plt.scatter(s_l02,gini_l02,label = "$\lambda$ = 0.2")
plt.scatter(s_l035,gini_l035,label ="$\lambda$ = 0.35")
plt.scatter(s_l05,gini_l05,label = "$\lambda$ = 0.5")
plt.legend()

s_l02_teo = np.linspace(0,0.1,100)
a,b_l02 = ab_teo(s_l02_teo,0.2,0)
G = (1/np.sqrt(np.pi))*gamma(b_l02+1/2)/gamma(b_l02+1)
plt.plot(s_l02_teo,G)

s_l035_teo = np.linspace(0,0.2,100)
a,b_l035 = ab_teo(s_l035_teo,0.35,0)
G = (1/np.sqrt(np.pi))*gamma(b_l035+1/2)/gamma(b_l035+1)
plt.plot(s_l035_teo,G)

s_l05_teo = np.linspace(0,0.3,100)
a,b_l05 = ab_teo(s_l05_teo,0.5,0)
G = (1/np.sqrt(np.pi))*gamma(b_l05+1/2)/gamma(b_l05+1)

plt.xlabel("Guardado de producción $s$", fontsize = 13)
plt.ylabel("Indice de Gini $G$", fontsize = 13)
plt.legend(fontsize = 12)
plt.xticks(fontsize =12)
plt.yticks(fontsize =12)
plt.plot(s_l05_teo,G)
plt.title("Indice de Gini", fontsize = 12)
plt.savefig("gini.pdf", bbox_inches='tight' )

plt.show()




##############################################################
# Simulación 3: simulación del caso para dos tipos de agentes:
# Trabajadores y productores
##############################################################

N =1000 #Numero de agentes N
ensembles = 10**3 #Numero de ensambles promediados
time_steps = 20100 #Pasos de tiempo considerados
histogram_bins = 100 #Numero de divisiones para los histogramas que general la densidad de probabilidad
view_times = 100
init_time_ab = 10000

l = [0.5]*4   #Valores de lambda
s = [0.1]*4  #Valores de s
alpha = [0.05,0.3,0.6,0.9]

#Varables de los productores
hist_p = []                       #Vector que almacena en cada entrada los histogramas de cada para de parametros lambda y s. Es el histograma "normalizado" con density = True en np.histogram
hist_norm_p = []                  #Vector con los histogramas con la suma normalizada a 1
total_wealth_p = []               #Vector con la evolución temporal de la riqueza de cada l y s en cada entrada
time = []                       #Vector que guarda el paso de timepo de los instantes de tiempo que se estan guardando
min_p = []                        #Vector que guarda la riqueza mínima en cada entrada para cada l y s escogidos
max_p = []                        #Vector que guarda la riqueza maxima en cada entrada para cada l y s escogidos

# variables de los trabajdores
hist_w = []                       
hist_norm_w = []                  
total_wealth_w = []              
min_w = []                        
max_w = []                        


for idxl, i in enumerate(l):
    result = probability_density_workers(N,alpha[idxl],0.1,s[idxl],l[idxl],ensembles,time_steps,histogram_bins, view_times)
    
    hist_p.append(result[0])
    hist_w.append(result[1])
    hist_norm_p.append(result[2])
    hist_norm_w.append(result[3])
    total_wealth_p.append(result[4])
    total_wealth_w.append(result[5])
    time.append(result[6])
    min_p.append(result[7])
    min_w.append(result[8])
    max_p.append(result[9])
    max_w.append(result[10])

##############################################################
#Grafica de las distribuciones de riqueza cuasi-estáticas de
# trabajadores y productores, así como el calculo del indice 
# de Gini en cada caso mostrado en los legend de las graficas 
##############################################################

for idx, i in enumerate(l):
    width_p = (max_p[idx]-min_p[idx])/histogram_bins 
    width_w = (max_w[idx]-min_w[idx])/histogram_bins 
    hist_x_p = np.linspace(min_p[idx]+width_p/2,max_p[idx]-width_p/2,histogram_bins)
    hist_x_w = np.linspace(min_w[idx]+width_w/2,max_w[idx]-width_w/2,histogram_bins)

    promedio_p = moment_r(1,hist_norm_p[idx][-1],min_p[idx],max_p[idx],histogram_bins)
    promedio_w = moment_r(1,hist_norm_w[idx][-1],min_w[idx],max_w[idx],histogram_bins)



    gini_p = gini_index(hist_norm_p[idx][-1],min_p[idx],max_p[idx],histogram_bins)
    gini_w = gini_index(hist_norm_w[idx][-1],min_p[idx],max_p[idx],histogram_bins)


    plt.plot(hist_x_p/promedio_p,hist_p[idx][-1]*promedio_p,label = "Productores. G = {:.3f}".format(gini_p) ,marker = ".",linestyle = "None")
    plt.plot(hist_x_w/promedio_w,hist_w[idx][-1]*promedio_w,label = "Trabajadores. G = {:.3f}".format(gini_w),marker = ".",linestyle = "None")


    x = np.linspace(0,2,1000)
    y_ajuste = gamma_dist(x/promedio_p,0.1/promedio_p,10)
    plt.legend(fontsize = 12)
    plt.xlim([0,np.minimum(max_p[idx]/promedio_p,max_w[idx]/promedio_w)])
    plt.xlabel("Nivel de riqueza  $w/\\langle w \\rangle$", fontsize = 13)
    plt.ylabel("Densidad de Probabilidad", fontsize = 13)  
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title("$\lambda$ = {}, $s$ = {}, $\\alpha$ = {}".format(l[idx],s[idx],alpha[idx]))
    plt.savefig("distribucion_workers_alpha_{}.pdf".format(alpha[idx]),bbox_inches="tight")
    plt.show()

##############################################################
#Grafica de la evolución temporal de la riqueza promedio de los
# productores 
##############################################################

for idx, i in enumerate(l):
    y_teo = wealth_teo_workers(s[idx],l[idx],alpha[idx],N*0.1,time[0])
    plt.plot(time[0],total_wealth_p[idx]/(N*0.1),linestyle = "None", marker = ".", markevery = 10, label = "$\\alpha$ = {}".format(alpha[idx]))
    if idx == 3:
        plt.plot(time[0],y_teo, color = "#637074", zorder = -1, label = "$\\langle w_p \\rangle = \\langle w_0 \\rangle\exp\\left(g_e\dfrac{2t}{N_p}\\right)$")
    else:
        plt.plot(time[0],y_teo, color = "#637074", zorder = -1)
plt.yscale("log")
plt.legend()
plt.xticks(range(0,20100,5000),fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Pasos de tiempo",fontsize=13)
plt.ylabel("Riqueza promedio de productores $\\langle$ $w_p$ $\\rangle$ ", fontsize = 13)
plt.title("Riqueza promedio productores $\\langle w_p \\rangle$: $\lambda$ = 0.5, $s$ = 0.1", fontsize = 12)
plt.savefig("riqueza_promedio_productores.pdf", bbox_inches='tight' )
plt.show()
