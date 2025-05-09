# De aqui a abajo podemos ir agregando funciones para el proyecto, los temás que tenemos son: Teoremas de taylor, (el resto vamos viendo porque no me acuerdo cuales son XD)
from math import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("""
        Bienvenido!! 
        Digite el tema que desea realizar.
            1- Polinomios de taylor
            2- Teoria del Error
            3- Binarios
            4- Metodo de newton
            5- Metodo Bisercion
""")
    choosen = int(input("Opcion: "))
    if choosen == 1:
        a = int(input ("Digite el punto alrededor del cual desea el polinomio (X0 = ?): "))
        n = int(input("Digite el orden del polinomio de taylor: "))
        taylor(a, n) #taylor recibe tanto un x0 como un Pn(x)
    else:
        print("Opcion no realizada todavia/ Opcion invalida")
        main()
        
def taylor(a, n):
    x = sp.symbols('x') #crea una variable simbolica
    expr = input("Digite la funcion f(x): ")
    f = sp.sympify(expr) #toma la funcion expresada en 'expr'
    F = f
    T = f.subs(x, a) #en f sustituye la variable x en el valor a
    for k in range(1, n+1): 
        dfk = sp.diff(f, x) #toma la derivada k-esima derivando la funcion f
        T = T+dfk.subs(x, a)*((x-a)**k)/factorial(k) #evaluamos la derviada k-esima en el punto a, esto lo multiplicamos y luego dividimos sobre el factorial de k
        f = dfk #se utiliza para optimizar un poco el codigo

    print("\nPolinomio de Taylor:")
    print(sp.expand(T)) # utilizado para expandir el polinomio de taylor, visualizandolo de mejor forma

    f_func = sp.lambdify(x, F, modules=["numpy"])
    t_func = sp.lambdify(x, T, modules=["numpy"])

    x_vals = np.linspace(a - 3, a + 3, 400)
    y_vals_f = f_func(x_vals)
    y_vals_t = t_func(x_vals)

    plt.plot(x_vals, y_vals_f, label='f(x)', color='blue') # grafica la funcion f(x) de color azul
    plt.plot(x_vals, y_vals_t, label=f'Taylor orden {n}', color='black', linestyle='--') # grafica el polinomio de taylor de orden n de color negro 
    plt.title('Polinomio de Taylor')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

    question = input("Desea volver al menu? (Si/No): ")
    if question == "si" or question == "Si":
        main()
    else:
        print("Adios")



if __name__ == "__main__":
    main()