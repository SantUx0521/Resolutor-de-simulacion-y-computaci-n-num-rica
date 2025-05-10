import streamlit as st
from math import factorial
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

st.title("¡Bienvenido!")
st.subheader("Selecciona el tema que deseas explorar:")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Polinomios de Taylor",
    "📈 Teoría del Error",
    "🔢 Binarios",
    "🍎 Método de Newton-Raphson",
    "✂️ Método de Bisección"
])

with tab1:
    st.header("Polinomios de Taylor")
    a = st.number_input("Digite el punto alrededor del cual desea el polinomio (X0): ", value=0.0, key="taylor_a")
    n = st.slider("Digite el orden del polinomio de Taylor:", min_value=1, max_value=10, value=3, key="taylor_n")
    expr = st.text_input("Digite la función f(x): ", value="sin(x)", key="taylor_fx")

    if st.button("Generar Polinomio de Taylor", key="taylor_generar"):
        x = sp.symbols('x')
        f_sym = sp.sympify(expr, locals={'e': sp.exp(1)})
        F = f_sym
        T = f_sym.subs(x, a)
        for k in range(1, n + 1):
            dfk = sp.diff(f_sym, x)
            T = T + dfk.subs(x, a) * ((x - a) ** k) / factorial(k)
            f_sym = dfk

        st.subheader("Polinomio de Taylor:")
        st.latex(sp.expand(T))

        f_func = sp.lambdify(x, F, modules=["numpy"])
        t_func = sp.lambdify(x, T, modules=["numpy"])

        x_vals = np.linspace(float(a) - 3, float(a) + 3, 400)
        y_vals_f = f_func(x_vals)
        y_vals_t = t_func(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals_f, label='f(x)', color='black')
        ax.plot(x_vals, y_vals_t, label=f'Taylor orden {n}', color='purple', linestyle='--')
        ax.set_title('Polinomio de Taylor')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Opciones Adicionales:")
        if st.checkbox("Mostrar error absoluto en un punto", key="taylor_error_abs_check"):
            x_eval_error = st.number_input("Valor de x para evaluar el error absoluto:", value=float(a) + 1, key="taylor_error_abs_x")
            error = abs(f_func(x_eval_error) - t_func(x_eval_error))
            st.write(f"f({x_eval_error}) = {f_func(x_eval_error)}")
            st.write(f"P{n}({x_eval_error}) = {t_func(x_eval_error)}")
            st.write(f"Error absoluto |f(x) - Pn(x)| = {error}")

        if st.checkbox("Mostrar cota del error máximo", key="taylor_cota_check"):
            x_eval_cota = st.number_input("Valor de x para evaluar la cota del error:", value=float(a) + 2, key="taylor_cota_x")
            df_n1 = f_sym
            for i in range(n + 1):
                df_n1 = sp.diff(df_n1, x)
            df_n1_func = sp.lambdify(x, abs(df_n1), modules=["numpy"])
            points = np.linspace(min(a, x_eval_cota), max(a, x_eval_cota), 1000)
            max_derivative = max(df_n1_func(points))
            remainder = (max_derivative * abs(x_eval_cota - a) ** (n + 1)) / factorial(n + 1)
            st.write(f"Cota del error máximo (resto de Taylor): {remainder}")

with tab2:
    st.header("Teoría del Error")
    st.subheader("Cálculo de Error Absoluto y Relativo")
    verdadero = st.number_input("Ingrese el valor verdadero [P]: ", value=0.0, key="error_verdadero")
    aproximado = st.number_input("Ingrese el valor de aproximación de p [P*]: ", value=0.0, key="error_aproximado")
    if st.button("Calcular Errores", key="error_calcular"):
        error_abs = abs(verdadero - aproximado)
        error_rel = error_abs / abs(verdadero) if verdadero != 0 else float('inf')
        st.write(f"Error absoluto: {error_abs}")
        st.write(f"Error relativo: {error_rel:.5f} ({error_rel * 100:.2f}%)")

    st.subheader("Aritmética con Redondeo/Corte")
    num1 = st.number_input("Ingrese el primer número [P]: ", value=0.0, key="aritmetica_num1")
    num2 = st.number_input("Ingrese el segundo número [q]: ", value=0.0, key="aritmetica_num2")
    digitos = st.number_input("Ingrese el número de dígitos significativos: ", min_value=1, step=1, value=3, key="aritmetica_digitos")
    metodo_aritmetica = st.radio("Método:", ["Redondeo", "Corte"], key="aritmetica_metodo")

    def procesar_sig(num, sig, metodo):
        if num == 0:
            return 0.0
        exp = int(np.floor(np.log10(abs(num))))
        factor = 10 ** (sig - exp - 1)
        if metodo == "Redondeo":
            resultado = round(num * factor) / factor
        else:
            resultado = np.floor(num * factor) / factor
        return resultado

    def format_sig(x, sig):
        if x == 0:
            return f"{0:.{sig-1}f}"
        else:
            return f"{x:.{sig - int(np.floor(np.log10(abs(x)))) - 1}f}"

    if st.button("Procesar Aritmética", key="aritmetica_procesar"):
        num1_proc = procesar_sig(num1, int(digitos), metodo_aritmetica)
        num2_proc = procesar_sig(num2, int(digitos), metodo_aritmetica)
        resta_exacta = num1 - num2
        resta_proc = num1_proc - num2_proc
        error_abs_op = abs(resta_exacta - resta_proc)
        error_rel_op = abs(resta_exacta - resta_proc) / abs(resta_exacta) if resta_exacta != 0 else 0

        st.write(f"{metodo_aritmetica} de {num1} a {digitos} dígitos: {format_sig(num1_proc, int(digitos))}")
        st.write(f"{metodo_aritmetica} de {num2} a {digitos} dígitos: {format_sig(num2_proc, int(digitos))}")
        st.write(f"Resultado exacto: {num1} - {num2} = {resta_exacta:.5f}")
        st.write(f"Resultado procesado: {format_sig(num1_proc, int(digitos))} - {format_sig(num2_proc, int(digitos))} = {format_sig(resta_proc, int(digitos))}")
        st.write(f"Error absoluto de la operación: {error_abs_op:.5f}")
        st.write(f"Error relativo de la operación: {error_rel_op:.5f} ({error_rel_op * 100:.2f}%)")

with tab3:
    st.header("Binarios")
    st.info("Aquí podríamos agregar funcionalidades para convertir entre sistemas numéricos o realizar operaciones binarias. ¡Esta sección está lista para que la desarrolles!")

with tab4:
    st.header("Método de Newton-Raphson")
    x_newton = sp.symbols('x')
    fx_input_newton = st.text_input("Ingrese la función f(x): ", value="x**2 - 2", key="newton_fx")
    x0_newton = st.number_input("Ingrese el valor inicial x0: ", value=1.5, key="newton_x0")
    tol_newton = st.number_input("Ingrese la tolerancia:", value=1e-5, key="newton_tol")
    imax_newton = st.number_input("Ingrese el número máximo de iteraciones:", min_value=1, step=1, value=50, key="newton_imax")

    if st.button("Calcular Raíz (Newton-Raphson)", key="newton_calcular"):
        locals_dict_newton = {'e': sp.E, 'ln': sp.log}
        fx_expr_newton = sp.sympify(fx_input_newton, locals=locals_dict_newton)
        dfx_expr_newton = sp.diff(fx_expr_newton, x_newton)
        f_newton = sp.lambdify(x_newton, fx_expr_newton, modules=[{'e': np.e}, 'numpy'])
        f1_newton = sp.lambdify(x_newton, dfx_expr_newton, modules=[{'e': np.e}, 'numpy'])

        xr = x0_newton
        ea = 2 * tol_newton
        i = 0
        tabla_newton = []
        tabla_newton.append([i, xr, f_newton(xr), f1_newton(xr), "--", "--"])

        while ea > tol_newton and i < imax_newton:
            x_old = xr
            xr = xr - f_newton(xr) / f1_newton(xr)
            i += 1
            ea = abs(xr - x_old)
            er = abs(ea / xr) * 100 if xr != 0 else 0
            tabla_newton.append([i, xr, f_newton(xr), f1_newton(xr), ea, er])

        st.subheader("Resultados del Método de Newton-Raphson")
        st.write(f"Raíz aproximada: x = {xr:.8f}, f(x) = {f_newton(xr):.8f}")
        st.dataframe(tabla_newton, column_config={"Iteración": st.column_config.NumberColumn(),
                                                    "x": st.column_config.NumberColumn(format="%.8f"),
                                                    "f(x)": st.column_config.NumberColumn(format="%.8f"),
                                                    "f'(x)": st.column_config.NumberColumn(format="%.8f"),
                                                    "Error abs": st.column_config.NumberColumn(format="%.8e"),
                                                    "Error rel (%)": st.column_config.NumberColumn(format="%.2f")})

        x_vals_newton = np.linspace(xr - 2, xr + 2, 400)
        y_vals_newton = f_newton(x_vals_newton)
        fig_newton, ax_newton = plt.subplots()
        ax_newton.axhline(0, color='gray', linestyle=':')
        ax_newton.plot(x_vals_newton, y_vals_newton, label="f(x)", color='blue')
        ax_newton.plot(xr, f_newton(xr), 'ro', label="Raíz aproximada")
        ax_newton.set_title("Método de Newton-Raphson")
        ax_newton.set_xlabel("x")
        ax_newton.set_ylabel("f(x)")
        ax_newton.legend()
        ax_newton.grid(True)
        st.pyplot(fig_newton)

with tab5:
    st.header("Método de Bisección")
    x_biseccion = sp.symbols('x')
    fx_input_biseccion = st.text_input("Ingrese la función f(x): ", value="x**2 - 2", key="biseccion_fx")
    a_biseccion = st.number_input("Ingrese el límite inferior a:", value=1.0, key="biseccion_a")
    b_biseccion = st.number_input("Ingrese el límite superior b:", value=2.0, key="biseccion_b")
    tol_biseccion = st.number_input("Ingrese la tolerancia:", value=1e-5, key="biseccion_tol")
    imax_biseccion = st.number_input("Ingrese el número máximo de iteraciones:", min_value=1, step=1, value=50, key="biseccion_imax")

    if st.button("Calcular Raíz (Bisección)", key="biseccion_calcular"):
        locals_dict_biseccion = {'e': sp.E, 'ln': sp.log}
        fx_expr_biseccion = sp.sympify(fx_input_biseccion, locals=locals_dict_biseccion)
        f_biseccion = sp.lambdify(x_biseccion, fx_expr_biseccion, modules=[{'e': np.e}, 'numpy'])

        if f_biseccion(a_biseccion) * f_biseccion(b_biseccion) > 0:
            st.error("Error: La función debe cambiar de signo en el intervalo [a, b].")
        else:
            xr = (a_biseccion + b_biseccion) / 2
            ea = np.inf
            i = 0
            tabla_biseccion = []
            tabla_biseccion.append([i, a_biseccion, b_biseccion, xr, f_biseccion(xr), "--"])

            while ea > tol_biseccion and i < imax_biseccion:
                xr_old = xr
                if f_biseccion(a_biseccion) * f_biseccion(xr) < 0:
                    b_biseccion = xr
                else:
                    a_biseccion = xr
                xr = (a_biseccion + b_biseccion) / 2
                i += 1
                ea = abs(xr - xr_old)
                tabla_biseccion.append([i, a_biseccion, b_biseccion, xr, f_biseccion(xr), ea])

            st.subheader("Resultados del Método de Bisección")
            st.write(f"Raíz aproximada: x = {xr:.8f}, f(x) = {f_biseccion(xr):.8f}")
            st.dataframe(tabla_biseccion, column_config={"Iteración": st.column_config.NumberColumn(),
                                                        "a": st.column_config.NumberColumn(format="%.8f"),
                                                        "b": st.column_config.NumberColumn(format="%.8f"),
                                                        "xr": st.column_config.NumberColumn(format="%.8f"),
                                                        "f(xr)": st.column_config.NumberColumn(format="%.8f"),
                                                        "Error abs": st.column_config.NumberColumn(format="%.8e")})

            x_vals_biseccion = np.linspace(min(a_biseccion, b_biseccion) - 1, max(a_biseccion, b_biseccion) + 1, 400)
            y_vals_biseccion = f_biseccion(x_vals_biseccion)
            fig_biseccion, ax_biseccion = plt.subplots()
            ax_biseccion.axhline(0, color='gray', linestyle=':')
            ax_biseccion.plot(x_vals_biseccion, y_vals_biseccion, label="f(x)", color='blue')
            ax_biseccion.plot(xr,f_biseccion(xr), 'ro', label="Raíz aproximada")
            ax_biseccion.set_title("Método de Bisección")
            ax_biseccion.set_xlabel("x")
            ax_biseccion.set_ylabel("f(x)")
            ax_biseccion.legend()
            ax_biseccion.grid(True)
            st.pyplot(fig_biseccion)

with tab3:
    st.header("Binarios")
