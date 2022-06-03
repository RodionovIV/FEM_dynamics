import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class Data:

    def __init__(self):
        self.__dim = None
        self.__poisson = None
        self.__youngModulus = None
        self.__ro = None
        self.__n_nodes_ = None
        self.__nodes = None
        self.__n_elements = None
        self.__elements = None
        self.__n_constraints = None
        self.__constraints = None
        self.__n_loads = None
        self.__loads = None
        self.__D_matrix = None

    @property
    def nodes(self):
        return self.__n_nodes, self.__nodes

    @property
    def elements(self):
        return self.__n_elements, self.__elements

    @property
    def constraints(self):
        return self.__n_constraints, self.__constraints

    @property
    def loads(self):
        return self.__n_loads, self.__loads

    @property
    def ro(self):
        return self.__ro

    @property
    def D(self):
        p,y = self.__poisson, self.__youngModulus
        self.__D_matrix = np.array([[1.,p,0.],[p,1.,0.],[0.,0.,(1 - p)/2]]) * (y/(1 - p**2))
        return self.__D_matrix

    @property
    def dim(self):
        return self.__dim

    @nodes.setter
    def nodes(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {nodes}')
        with open(filename,'r') as nodes_file:
            self.__n_nodes = int(nodes_file.readline())
            self.__nodes = np.array([list(map(float, line.split())) for line in nodes_file])

    @elements.setter
    def elements(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {elements}')
        with open(filename,'r') as elements_file:
            self.__n_elements = int(elements_file.readline())
            self.__elements = np.array([list(map(int, line.split())) for line in elements_file])


    @constraints.setter
    def constraints(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {constraints}')
        with open(filename,'r') as constraints_file:
            self.__n_constraints = int(constraints_file.readline())
            self.__constraints = np.array([list(map(int, line.split())) for line in constraints_file])

    @loads.setter
    def loads(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {loads}')
        if filename == 'stress.txt':
            elements, nodes = self.__elements, self.__nodes
            dot = lambda v1, v2: v1[0]*v2[0] + v1[1]*v2[1]
            l = lambda v: dot(v,v)**0.5
            stress = open(filename,'r')
            stress_count = int(stress.readline())
            loads = open('loads.txt','w')
            loads.write(f'{stress_count}')
            #loads.write(str(stress_count))
            completed = dict()
            for line in stress:
                e1,e2,element, nx, ny, P = list(map(float, line.split()))
                e1,e2,element = map(int,[e1,e2,element])
                e3 = elements[element][2]
                nx, ny = nx/l((nx,ny)), ny/l((nx,ny))
                x1, y1 = nodes[e1]
                x2, y2 = nodes[e2]
                x3, y3 = nodes[e3]
                check_x, check_y = x3 - x1, y3 - y1
                lx, ly = x2 - x1, y2 - y1

                cos = dot((nx,ny), (check_x,check_y))

                if cos > 0:
                    nx, ny = (-1)*nx, (-1)*ny
                L = l((lx, ly))


                if e1 in completed.keys():
                    completed[e1] = np.array([nx,ny])
                else:
                    completed[e1] = np.array([nx,ny])

                if e2 in completed.keys():
                    completed[e2] = np.array([nx,ny])
                else:
                    completed[e2] = np.array([nx,ny])
            for key, value in zip(completed.keys(), completed.values()):
                     loads.write(f'\n{key} {(-0.5)*value[0]*P*L} {(-0.5)*value[1]*P*L}')
            stress.close()
            loads.close()
            filename = 'loads.txt'

        with open(filename,'r') as loads_file:
            dim = self.__dim
            self.__n_loads = loads_count = int(loads_file.readline())
            self.__loads = np.zeros(dim * self.__n_nodes)
            for line in loads_file:
                node, load_x, load_y = [float(x) for x in line.split()]
                self.__loads[dim*int(node) + 0], self.__loads[dim*int(node) + 1] = load_x, load_y


    def make(self, filename=None):
        if filename == None:
            raise FileExistsError('Filename not specified! {config}')
        with open(filename,'r') as config_file:
            files = [line.split('\n')[0] for line in config_file]
            self.nodes, self.elements, self.constraints, self.loads = files


    def set_params(self, dim = None ,poisson = None, youngModulus = None ,set_default = False):
        if (dim is None) or (poisson is None) or (youngModulus is None) or (set_default == True):
            self.__dim = 2
            self.__poisson = 0.25
            self.__youngModulus = 2e+7
            self.__ro = 2400
        else:
            self.__dim = dim
            self.__poisson = poisson
            self.__youngModulus = youngModulus
            self.__ro = ro

    def get_params(self):
        return {'dim':self.__dim, 'poisson':self.__poisson, 'youngModulus':self.__youngModulus, 'ro':self.__ro}

    def info(self):
        print(f'dim: {self.__dim}\npoisson: {self.__poisson}\nyoungModulus: {self.__youngModulus}')


    def get_nodes(self):
        return self.__n_nodes, self.__nodes
    def get_elements(self):
        return self.__n_elements, self.__elements
    def get_constraints(self):
        return self.__n_constraints, self.__constraints
    def get_loads(self):
        return self.__n_loads, self.__loads
    def get_D(self):
        return self.__D_matrix

#Класс, отвечающий за решение задачи
class Solver:

    def __init__(self):
        self.__disp = None
        self.__eps = None
        self.__sigma = None
        self.__sigma_mises = None


#Основная вычислительная функция
    def process_dinamic(self, data):
        #Выгружаем данные
        n_nodes, nodes = data.nodes
        n_elements, elements = data.elements
        n_constraints, constraints = data.constraints
        n_loads, loads = data.loads
        D = data.D
        dim = data.dim
        ro = data.ro

        #Собираем глобальные матрицы К и М
        nodes_x, nodes_y = nodes.T
        K_rows, K_cols, K_values = [], [], []
        M_rows, M_cols, M_values = [], [], []

        for element in elements:

            x = np.array([nodes_x[element[0]], nodes_x[element[1]], nodes_x[element[2]]])
            y = np.array([nodes_y[element[0]], nodes_y[element[1]], nodes_y[element[2]]])
            X0, X1, X2 = nodes_x[element[0]],nodes_x[element[1]],nodes_x[element[2]]
            Y0, Y1, Y2 = nodes_y[element[0]],nodes_y[element[1]],nodes_y[element[2]]
            area = 0.5 * abs((X0 - X2) * (Y1 - Y2) - (X1 - X2) * (Y0 - Y2));
            mass = ro * area
            loc_M = np.eye(6) * mass/3

            C = np.matrix([np.ones(3),x,y])
            C = C.T
            IC = C.I

            B = np.zeros((3,6))

            for i in range(3):
                B[0, 2 * i + 0] = IC[1,i]
                B[1, 2 * i + 1] = IC[2,i]
                B[2, 2 * i + 0] = IC[2,i]
                B[2, 2 * i + 1] = IC[1,i]

            loc_K = np.matmul(np.matmul(B.T, D), B) * (np.linalg.det(C) / 2)

            for i in range(3):
                for j in range(3):
                        K_rows.extend([2 * element[i] + 0, 2 * element[i] + 0, 2 * element[i] + 1, 2 * element[i] +1])
                        K_cols.extend([2 * element[j] + 0, 2 * element[j] + 1, 2 * element[j] + 0, 2 * element[j] +1])
                        K_values.extend([loc_K[2*i + 0, 2*j + 0],loc_K[2*i + 0, 2*j + 1],loc_K[2*i + 1, 2*j + 0],loc_K[2*i + 1, 2*j + 1]])

                        M_rows.extend([2 * element[i] + 0, 2 * element[i] + 0, 2 * element[i] + 1, 2 * element[i] +1])
                        M_cols.extend([2 * element[j] + 0, 2 * element[j] + 1, 2 * element[j] + 0, 2 * element[j] +1])
                        M_values.extend([loc_M[2*i + 0, 2*j + 0],loc_M[2*i + 0, 2*j + 1],loc_M[2*i + 1, 2*j + 0],loc_M[2*i + 1, 2*j + 1]])


        #Преобразовываем матрицы K,М суммируем значения по индексам
        K_rows = np.array(K_rows)
        K_cols = np.array(K_cols)
        K_values = np.array(K_values)
        coord=np.vstack((K_rows, K_cols))
        u, indices = np.unique(coord, return_inverse=True, axis=1)
        K_values=np.bincount(indices, weights=K_values)
        K_rows, K_cols=np.vsplit(u, 2)
        N = dim * n_nodes
        K = np.array([K_rows[0], K_cols[0], K_values]).T

        M_rows = np.array(M_rows)
        M_cols = np.array(M_cols)
        M_values = np.array(M_values)
        coord=np.vstack((M_rows, M_cols))
        u, indices = np.unique(coord, return_inverse=True, axis=1)
        M_values=np.bincount(indices, weights=M_values)
        M_rows, M_cols=np.vsplit(u, 2)
        N = dim * n_nodes
        M = np.array([M_rows[0], M_cols[0], M_values]).T

        #Применяем закрепления (ApplyConstraints)
        ind2Constraint = [2 * i[0] + 0 for i in constraints if i[1] == 1 or i[1] == 3] + [2 * i[0] + 1 for i in constraints if i[1] == 2 or i[1] == 3]
        #diag_K = np.array([line for line in K if line[0] == line[1]])
        diag_K = [line for line in K if line[0] == line[1]]
        other_K = np.array([line for line in K if line[0] != line[1]])
        #nn_K = np.array([[line[0], line[1], 1] if line[0] in ind2Constraint else line for line in diag_K] + [[line[0],line[1],0] if line[0] in ind2Constraint or line[1] in ind2Constraint else line for line in other_K]).T
        nn_K = np.array(diag_K + [[line[0],line[1],0] if line[0] in ind2Constraint or line[1] in ind2Constraint else line for line in other_K]).T
        rows,cols,values = nn_K
        K =  csr_matrix((values, (rows, cols)), shape = (N,N))
        M =  csr_matrix((M_values, (M_rows[0], M_cols[0])), shape = (N,N))
        disp = spsolve(K, loads).reshape((-1,dim))

        #Считаем собственные значения и собственные векторы
        evals, evectors = calc_eigenvalues(K,M)
        evectors_T = evectors.T
        # for i,evector in enumerate(evectors_T):
        #     U_sum = [(disp[0]**2 + disp[1]**2)**0.5 for disp in evector.reshape((-1,dim))]
        #     visual_form(U_sum,data,i)
        print('Calculate')
        a = 10
        fixed_x, fixed_y = 4., 2.
        sigma_xx_data, sigma_yy_data, sigma_xy_data, time_data = [], [], [], []
        nt = 1
        dt = 0.02
        end_time = 1
        end_nt = end_time // dt
        while nt <= end_nt:
            U_final = 0
            t = nt*dt
            for evector, eval in zip(evectors_T, evals):
                f_i = evector.T @ loads * np.cos(a*t)
                m_i = evector.T @ M @ evector
                y_i0 = evector.T @ M @ evectors_T[0]
                y_i = y_i0*np.cos(eval**(0.5) * t) + f_i / (m_i * (eval - a**2))
                U_final = U_final + y_i * evector.T
                #print(evector.shape)
                #print(U_final[:2])
            #print('Tut')
            U_final = U_final.reshape((-1,dim))
            sigma_xx,sigma_yy,sigma_xy = sigma_calculate(U_final,data,nt)
            sigma_xx_value, sigma_yy_value, sigma_xy_value = interpolateSigma(fixed_x,fixed_y, sigma_xx, nodes, elements), interpolateSigma(fixed_x,fixed_y, sigma_yy, nodes, elements),interpolateSigma(fixed_x,fixed_y, sigma_xy, nodes, elements)
            #print('****')
            #print(sigma_xx_value)
            sigma_xx_data.append(sigma_xx_value)
            sigma_yy_data.append(sigma_yy_value)
            sigma_xy_data.append(sigma_xy_value)
            time_data.append(t)
            nt+=1
            #if abs(max(disp) - max(U_final)) <= 1e-3:
            #    print(f'Norm:{abs(max(disp) - max(U_final))}')
            #    print(nt)
            #norm = np.linalg.norm(disp-U_final)
            #print(nt, norm)
            #if pred_norm < norm:
            #    break
            #pred_norm = norm
        with open('sigma_file.txt','w') as s_file:
            for t, xx, yy, xy in zip(time_data, sigma_xx_data, sigma_yy_data, sigma_xy_data):
                s_file.write(f'{t} {xx} {yy} {xy}\n')

        U_final = U_final.reshape((-1,dim))
        sigma_xx,sigma_yy,sigma_xy = sigma_calculate(U_final,data,nt)
        return U_final, sigma_xx, sigma_yy, sigma_xy



def visual_form(disp, data, number):
    n_nodes, nodes = data.nodes
    n_elements, elements = data.elements
    nodes_x, nodes_y = nodes.T

    size = (10, 5)
    plt.figure(figsize = size)
    plt.title(f'MOD {number+1}', fontsize = 15)
    plt.tripcolor(nodes_x, nodes_y, elements, disp)
    plt.colorbar()
    plt.savefig(f'MOD {number+1}.png')
    plt.close()


def sigma_calculate(disp,data,nt):
    n_nodes, nodes = data.nodes
    n_elements, elements = data.elements
    n_constraints, constraints = data.constraints
    n_loads, loads = data.loads
    D = data.D
    dim = data.dim
    ro = data.ro

    #Собираем глобальную матрицу жесткости
    #CalculateStiffnessMatrix{
    nodes_x, nodes_y = nodes.T

    sigma = []
    sigma_mises = []
    eps = []

    for element in elements:

        x = np.array([nodes_x[element[0]], nodes_x[element[1]], nodes_x[element[2]]])
        y = np.array([nodes_y[element[0]], nodes_y[element[1]], nodes_y[element[2]]])

        C = np.matrix([np.ones(3),x,y])
        C = C.T
        IC = C.I

        B = np.zeros((3,6))

        for i in range(3):
            B[0, 2 * i + 0] = IC[1,i]
            B[1, 2 * i + 1] = IC[2,i]
            B[2, 2 * i + 0] = IC[2,i]
            B[2, 2 * i + 1] = IC[1,i]

        delta = np.vstack([disp[element[0]], disp[element[1]], disp[element[2]]]).reshape((-1,1))
        eps.extend([np.matmul(B,delta)])
        sigma.extend([np.matmul(D,eps[-1]).reshape((1,-1))])
        #sigma_mises.extend([mises(sigma[-1])])


    n_elements, elements = data.elements
    n_nodes, nodes = data.nodes

    sigma_xx = [j[0][0] for j in sigma]
    sigma_yy = [j[0][1] for j in sigma]
    sigma_xy = [j[0][2] for j in sigma]
    nodes_x, nodes_y = nodes.T

    #return sigma_xx, sigma_yy, sigma_xy
    #
    # size = (20, 10)
    # os.chdir('./Sigma_XX')
    # plt.figure(figsize = size)
    # plt.title('Sigma XX', fontsize = 15)
    # plt.tripcolor(nodes_x, nodes_y, elements, sigma_xx)
    # plt.colorbar()
    # plt.savefig(f'{nt}_XX.png')
    # plt.close()
    # os.chdir('..')
    #
    # os.chdir('./Sigma_YY')
    # plt.figure(figsize = size)
    # plt.title('Sigma YY', fontsize = 15)
    # plt.tripcolor(nodes_x, nodes_y, elements, sigma_yy)
    # plt.colorbar()
    # plt.savefig(f'{nt}_YY.png')
    # plt.close()
    # os.chdir('..')
    #
    # os.chdir('./Sigma_XY')
    # plt.figure(figsize = size)
    # plt.title('Sigma XY', fontsize = 15)
    # plt.tripcolor(nodes_x, nodes_y, elements, sigma_xy)
    # plt.colorbar()
    # plt.savefig(f'{nt}_XY.png')
    # plt.close()
    # os.chdir('..')

    return sigma_xx, sigma_yy, sigma_xy

def calc_eigenvalues (K, M, n=10):

    evals, evecs =  sc.sparse.linalg.eigsh(K,n,M,which ='SM')

    print((abs(evals)**0.5)/(2*np.pi))
    return evals, evecs



def in_triangle(x,y,element, nodes):
    x0, y0 = x,y
    x1,y1 = nodes[element[0]]
    x2,y2 = nodes[element[1]]
    x3,y3 = nodes[element[2]]

    r1 = (x1 - x0) * (y2 - y1) - (x2 - x1) * (y1 - y0)
    r2 = (x2 - x0) * (y3 - y2) - (x3 - x2) * (y2 - y0)
    r3 = (x3 - x0) * (y1 - y3) - (x1 - x3) * (y3 - y0)

    if (r1 <= 0 and r2 <= 0 and r3 <= 0) or (r1 >= 0 and r2 >= 0 and r3 >= 0):
        return True
    else:
        return False


def interpolateSigma(x, y, sigma, nodes, elements):
    s = 0
    for key, val in enumerate(elements):
            if in_triangle(x,y,val,nodes):
                #s = sigma[key][0][axis]
                s = sigma[key]
                return s
