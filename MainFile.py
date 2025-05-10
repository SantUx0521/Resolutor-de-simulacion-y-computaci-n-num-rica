# De aqui a abajo podemos ir agregando funciones para el proyecto, los temás que tenemos son: Teoremas de taylor, (el resto vamos viendo porque no me acuerdo cuales son XD)
from math import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def main():
    while True:
        print("""
            Bienvenido!! 
            Digite el tema que desea realizar.
                1- Polinomios de taylor
                2- Teoria del Error
                3- Binarios
                4- Metodo de Newton Raphson
                5- Metodo de Bisercion
                0 - Salir
    """)
        choosen = int(input("Opcion: "))
        if choosen == 1:
            a = int(input ("Digite el punto alrededor del cual desea el polinomio (X0 = ?): "))
            n = int(input("Digite el orden del polinomio de taylor: "))
            taylor(a, n) #taylor recibe tanto un x0 como un Pn(x)
        elif choosen == 2:
            teoria_del_error()
        
        elif choosen == 3:
            binario()
        
        elif choosen == 4:
            NewtonRaphson()
        
        elif choosen == 5:
            MetodoBisec()

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
            sub_option  = int(input("Opción: "))

            if sub_option  == 1:
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

            elif sub_option  == 2:
                x_eval = float(input("Valor de x para evaluar el error absoluto: "))
                f_func = sp.lambdify(x, F, modules=["numpy"])  # usa F, no f
                t_func = sp.lambdify(x, T, modules=["numpy"])
                error = abs(f_func(x_eval) - t_func(x_eval))
                print(f"\nf({x_eval}) = {f_func(x_eval)}")
                print(f"P{n}({x_eval}) = {t_func(x_eval)}")
                print(f"\nError absoluto |f(x) - Pn(x)| = {error}")

            elif sub_option  == 3:
                x_eval = float(input("Valor de x para evaluar la cota del error: "))
                df_n1 = f
                for i in range(n+1):
                    df_n1 = sp.diff(df_n1, x)
                df_n1_func = sp.lambdify(x, abs(df_n1), modules=["numpy"])
                points = np.linspace(min(a, x_eval), max(a, x_eval), 1000)
                max_derivative = max(df_n1_func(points))
                remainder  = ( max_derivative * abs(x_eval - a)**(n+1)) / factorial(n+1)
                print(f"\nCota del error máximo (resto de Taylor): {remainder }")

            elif sub_option  == 0:
                break
            else:
                print("Opción inválida.")

def teoria_del_error():
    while True:
        print("""
        --- Submenú de Teoría del Error ---
            1 - Calcular error absoluto y relativo
            2 - Aritmética con redondeo y su error
            3 - Aritmética con corte y su error
            0 - Volver al menú principal
        """)
        option = int(input("Opción: "))

        if option == 1:
            verdadero = float(input("Ingrese el valor verdadero [P]: "))
            aproximado = float(input("Ingrese el valor de aproximación de p [P*]: "))
            error_abs = abs(verdadero - aproximado)
            error_rel = error_abs / abs(verdadero) if verdadero != 0 else float('inf')
            print(f"\nError absoluto: {error_abs}")
            print(f"Error relativo: {error_rel:.5f} ({error_rel * 100:.2f}%)")

        elif option == 2 or option == 3:
            num1 = float(input("Ingrese el primer número [P]: "))
            num2 = float(input("Ingrese el segundo número [q]: "))
            digitos = int(input("Ingrese el número de dígitos significativos: "))

            def procesar(num, metodo):
                if num == 0:
                    return 0.0
                exp = int(np.floor(np.log10(abs(num))))
                factor = 10 ** (digitos - exp - 1)
                if metodo == "redondeo":
                    resultado = round(num * factor) / factor
                else:
                    resultado = np.floor(num * factor) / factor
                return resultado

            metodo = "redondeo" if option == 2 else "corte"

            num1_proc = procesar(num1, metodo)
            num2_proc = procesar(num2, metodo)

            def format_sig(x, sig):
                if x == 0:
                    return f"{0:.{sig-1}f}"
                else:
                    return f"{x:.{sig - int(np.floor(np.log10(abs(x)))) - 1}f}"

            print(f"\n{metodo.capitalize()} de {num1} a {digitos} dígitos significativos: {format_sig(num1_proc, digitos)}")
            print(f"{metodo.capitalize()} de {num2} a {digitos} dígitos significativos: {format_sig(num2_proc, digitos)}")

            # Operación exacta r = p - q
            resta_exacta = num1 - num2

            # Operación procesada r* = p* - q*
            resta_proc = num1_proc - num2_proc

            print(f"\nResultados de la aritmética ({metodo}):")
            print(f"\nr = {num1} - {num2} = {resta_exacta:.5f}")
            print(f"r* = {format_sig(num1_proc, digitos)} - {format_sig(num2_proc, digitos)} = {format_sig(resta_proc, digitos)}")

            error_abs_op = abs(resta_exacta - resta_proc)
            error_rel_op = abs(resta_exacta - resta_proc) / abs(resta_exacta) if resta_exacta != 0 else 0

            print(f"Error absoluto: |{resta_exacta:.5f} - {resta_proc:.5f}| = {error_abs_op:.5f}")
            print(f"Error relativo: {error_rel_op:.5f} ({error_rel_op * 100:.2f}%)\n")

        elif option == 0:
            break
        else:
            print("Opción inválida.")
def NewtonRaphson():
    # Solicita la función como texto
    x = sp.symbols('x')
    locals_dict = {'e': sp.E, 'ln': sp.log}
    fx_input = input("Ingresa la función f(x): ")
    fx_expr = sp.sympify(fx_input, locals=locals_dict)
    dfx_expr = sp.diff(fx_expr, x)

    # Crea funciones evaluables para f y f'
    f = sp.lambdify(x, fx_expr, modules=[{'e': np.e}, 'numpy'])
    f1 = sp.lambdify(x, dfx_expr, modules=[{'e': np.e}, 'numpy'])

    x0 = float(input("Ingresa el valor inicial x0: "))
    tol = float(input("Ingresa la tolerancia (ej. 1e-5): "))
    imax = int(input("Ingresa el número máximo de iteraciones: "))

    # Inicialización
    xr = x0
    ea = 2 * tol
    i = 0
    tabla = []
    tabla.append([i, xr, f(xr), f1(xr), "--", "--"])

    while ea > tol and i < imax:
        x_old = xr
        xr = xr - f(xr) / f1(xr)
        i += 1
        ea = abs(xr - x_old)
        er = abs(ea / xr) * 100 if xr != 0 else 0
        tabla.append([i, xr, f(xr), f1(xr), ea, er])

    # Mostrar resultados
    print("\nMétodo de Newton-Raphson")
    print(f"Raíz aproximada: x = {xr}, f(x) = {f(xr)}\n")
    print(tabulate(tabla, headers=["Iteración", "x", "f(x)", "f'(x)", "Error abs", "Error rel (%)"]))

    x_vals = np.linspace(xr - 5, xr + 5, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='gray', linestyle=':')
    plt.plot(x_vals, y_vals, label="f(x)", color='blue')
    plt.plot(xr, f(xr), 'ro', label="Raíz aproximada")
    plt.title("Método de Newton-Raphson")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

def MetodoBisec():
    # Solicita la función como texto
    x = sp.symbols('x')
    locals_dict = {'e': sp.E, 'ln': sp.log}
    fx_input = input("Ingresa la función f(x): ")
    fx_expr = sp.sympify(fx_input, locals=locals_dict)
    
    # Crea función evaluable
    f = sp.lambdify(x, fx_expr, modules=[{'e': np.e}, 'numpy'])
    
    # Pide los datos necesarios
    a = float(input("Ingrese el límite inferior a: "))
    b = float(input("Ingrese el límite superior b: "))
    tol = float(input("Ingrese la tolerancia (ej. 1e-5): "))
    imax = int(input("Ingrese el número máximo de iteraciones: "))

    if f(a) * f(b) > 0:
        print("Error: La función debe cambiar de signo en el intervalo [a, b].")
        return

    xr = (a + b) / 2
    ea = np.inf
    i = 0
    tabla = []
    tabla.append([i, a, b, xr, f(xr), "--"])

    while ea > tol and i < imax:
        xr_old = xr
        if f(a) * f(xr) < 0:
            b = xr
        else:
            a = xr
        xr = (a + b) / 2
        i += 1
        ea = abs(xr - xr_old)
        tabla.append([i, a, b, xr, f(xr), ea])

    # Mostrar resultados
    print("\nMétodo de Bisección")
    print(f"Raíz aproximada: x = {xr}, f(x) = {f(xr)}\n")

    print(tabulate(tabla, headers=["Iteración", "a", "b", "xr", "f(xr)", "Error abs"]))


    x_vals = np.linspace(a - 1, b + 1, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='gray', linestyle=':')
    plt.plot(x_vals, y_vals, label="f(x)", color='blue')
    plt.plot(xr, f(xr), 'ro', label="Raíz aproximada")
    plt.title("Método de Bisección")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
