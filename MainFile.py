# De aqui a abajo podemos ir agregando funciones para el proyecto, los temás que tenemos son: Teoremas de taylor, (el resto vamos viendo porque no me acuerdo cuales son XD)
from math import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def main():
    while True:
        print("""
            Bienvenido!! 
            Digite el tema que desea realizar.
                1- Polinomios de taylor
                2- Teoria del Error
                3- Binarios
                4- Metodo de newton
                5- Metodo Bisercion
                0 - Salir
    """)
        choosen = int(input("Opcion: "))
        if choosen == 1:
            a = int(input ("Digite el punto alrededor del cual desea el polinomio (X0 = ?): "))
            n = int(input("Digite el orden del polinomio de taylor: "))
            taylor(a, n) #taylor recibe tanto un x0 como un Pn(x)
        elif choosen == 0:
                    print("Saliendo del programa...")
                    break
        else:
            print("Opción inválida.")
        

def taylor(a, n):
    x = sp.symbols('x') #crea una variable simbolica
    expr = input("Digite la funcion f(x): ")
    f = sp.sympify(expr, locals={'e': sp.exp(1)}) #toma la funcion expresada en 'expr'
    F = f
    T = f.subs(x, a) #en f sustituye la variable x en el valor a
    for k in range(1, n+1): 
        dfk = sp.diff(f, x) #toma la derivada k-esima derivando la funcion f
        T = T+dfk.subs(x, a)*((x-a)**k)/factorial(k) #evaluamos la derviada k-esima en el punto a, esto lo multiplicamos y luego dividimos sobre el factorial de k
        f = dfk #se utiliza para optimizar un poco el codigo

    while True:
            print("""
            --- Submenú de Polinomios de Taylor ---
                1 - Mostrar polinomio de Taylor y graficar
                2 - Calcular error absoluto |f(x) - Pn(x)|
                3 - Calcular cota del error máximo (resto de Taylor)
                0 - Volver al menú principal
            """)
            sub_opcion = int(input("Opción: "))

            if sub_opcion == 1:
                print("\nPolinomio de Taylor:")
                print(sp.expand(T)) # utilizado para expandir el polinomio de taylor, visualizandolo de mejor forma

                f_func = sp.lambdify(x, F, modules=["numpy"])
                t_func = sp.lambdify(x, T, modules=["numpy"])

                x_vals = np.linspace(a - 3, a + 3, 400)
                y_vals_f = f_func(x_vals)
                y_vals_t = t_func(x_vals)
                
                mask = np.isreal(y_vals_f) & np.isreal(y_vals_t)
                x_plot = x_vals[mask]
                y_plot_f = np.real(y_vals_f[mask])
                y_plot_t = np.real(y_vals_t[mask])
                
                plt.plot(x_plot, y_plot_f, label='f(x)', color='black') # grafica la funcion f(x) de color azul
                plt.plot(x_plot, y_plot_t, label=f'Taylor orden {n}', color='purple', linestyle='--') # grafica el polinomio de taylor de orden n de color negro 
                plt.title('Polinomio de Taylor')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.grid(True)
                plt.legend()
                plt.show()

            elif sub_opcion == 2:
                x_eval = float(input("Valor de x para evaluar el error absoluto: "))
                f_func = sp.lambdify(x, F, modules=["numpy"])  # usa F, no f
                t_func = sp.lambdify(x, T, modules=["numpy"])
                error = abs(f_func(x_eval) - t_func(x_eval))
                print(f"\nError absoluto |f(x) - Pn(x)| = {error}")

            elif sub_opcion == 3:
                x_eval = float(input("Valor de x para evaluar la cota del error: "))
                df_n1 = f
                for i in range(n+1):
                    df_n1 = sp.diff(df_n1, x)
                df_n1_func = sp.lambdify(x, abs(df_n1), modules=["numpy"])
                puntos = np.linspace(min(a, x_eval), max(a, x_eval), 1000)
                max_derivada = max(df_n1_func(puntos))
                resto = (max_derivada * abs(x_eval - a)**(n+1)) / factorial(n+1)
                print(f"\nCota del error máximo (resto de Taylor): {resto}")

            elif sub_opcion == 0:
                break
            else:
                print("Opción inválida.")


if __name__ == "__main__":
    main()
