import streamlit as st
from math import factorial
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

st.title("¡Bienvenido!")
st.subheader("Selecciona el tema que deseas explorar:")

tab1, tab2, tab3, tab4, tab5= st.tabs([
    "📊 Polinomios de Taylor",
    "📈 Teoría del Error",
    "🔢 Bin",
    "🍎 Método de Newton-Raphson",
    "✂️ Método de Bisección",
])

tab6, tab7, tab8, tab9  = st.tabs([
    "📐 Interpolación de Newton",
    "🧷 Interpolación de Lagrange y Error",
    "❗Taylor con una aproximacion 10⁻ⁿ",
    "🧮 Diferencias divididas"
])

with tab1:
    st.header("Polinomios de Taylor")
    a = st.number_input("Digite el punto alrededor del cual desea el polinomio (X0): ", value=0.0, key="taylor_a")
    n = st.slider("Digite el orden del polinomio de Taylor:", min_value=1, max_value=10, value=3, key="taylor_n")
    expr = st.text_input("Digite la función f(x): ", value="sin(x)", key="taylor_fx")

    if st.button("Generar Polinomio de Taylor", key="taylor_generar"):
        x = sp.symbols('x')
        f_sym = sp.sympify(expr, locals={'e': sp.exp(1)})
        T = f_sym.subs(x, a)
        dfk = f_sym
        for k in range(1, n + 1):
            dfk = sp.diff(dfk, x)
            term = dfk.subs(x, a) * (x - a) ** k / factorial(k)
            T += term
            
        st.subheader("Polinomio de Taylor:")
        terms_latex = []
        current_df = f_sym
        for k in range(n + 1):
            if k == 0:
                coef = current_df.subs(x, a)
                terms_latex.append(f"{sp.latex(coef)}")
            else:
                current_df = sp.diff(current_df, x)
                deriv_eval = current_df.subs(x, a)
                coef = deriv_eval / factorial(k)

                if coef == 0:
                    continue  # Omitir términos nulos

                # Convertir a fracción racional para evaluar
                rat = sp.Rational(coef).limit_denominator()
                if rat.q == 1:
                    coef_latex = sp.latex(rat.p)
                else:
                    coef_latex = f"\\frac{{{rat.p}}}{{{rat.q}}}"

                # Manejar potencia
                if a == 0:
                    power = f"x^{k}" if k > 1 else "x"
                else:
                    power = f"(x - {a})^{k}" if k > 1 else f"(x - {a})"

                term = f"{coef_latex}{power}"
                terms_latex.append(term)

        taylor_str = " + ".join(terms_latex)
        st.latex(f"T_{{{n}}}(x) = {taylor_str}")

        f_func = sp.lambdify(x, dfk, modules=["numpy"])
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

with tab6:
    st.header("Interpolación de Newton")

 # Opción de entrada: sumatoria o valores directos
    modo = st.radio(
        "Selecciona el formato de entrada:",
        ["Sumatoria g(k)", "Valores Δ⁰ (f(n))"],
        index=0
    )

    # --- Modo 1: desde la sumatoria g(k)
    if modo == "Sumatoria g(k)":
        with st.expander("Modo 1: Desde sumatoria g(k)", expanded=True):
            # Formulario para ingresar g(k)
            with st.form("form_sumatoria"):
                st.markdown("Define $$f(n) = \sum_{k=1}^n g(k)$$ introduciendo **g(k)**:")
                g_expr = st.text_input("g(k) =", value="(2*n-1)**2", key="sum_g")
                submit_sum = st.form_submit_button("Calcular sumatoria")

            if submit_sum:
                import sympy as sp
                import pandas as pd

                # Definir símbolos
                n, k, x = sp.symbols('n k x')
                # Convertir la cadena g_expr a función simbólica
                g_sym = sp.sympify(g_expr, locals={'n': k, 'k': k})

                # Calcular valores de f(1), f(2), ... hasta detectar ceros en diferencias
                f_vals, ns = [], []
                m, orden_fin = 0, None
                while True:
                    m += 1
                    ns.append(m)
                    # f(m) = sum g_sym(k) desde k=1 hasta m
                    f_vals.append(sp.summation(g_sym, (k, 1, m)))

                    # Construcción provisional de la tabla de diferencias
                    dd = [[None]*m for _ in range(m)]
                    for i in range(m):
                        dd[i][0] = f_vals[i]
                    for j in range(1, m):
                        for i in range(m - j):
                            dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (ns[i+j] - ns[i])

                    # Buscar primera columna j donde Δ^j f(1) sea cero
                    for j in range(1, m):
                        if sp.simplify(dd[0][j]) == 0:
                            orden_fin = j
                            break
                    if orden_fin is not None:
                        break

                # Extraer coeficientes a_j y construir P(x)
                a = [dd[0][j] for j in range(orden_fin)]
                P = a[0]
                term = 1
                for j in range(1, orden_fin):
                    term *= (x - ns[j-1])  # factor (x - n_{j-1})
                    P += a[j] * term
                P_simpl = sp.simplify(P)

                # Mostrar resultados
                st.write(f"La tabla termina en Δ^{orden_fin}, con m = {m} puntos.")
                st.subheader("Valores de f(n):")
                st.table(list(zip(ns, [int(v) for v in f_vals])))

                st.subheader("Tabla de diferencias divididas:")
                cols = [f"Δ^{j}" for j in range(orden_fin+1)]
                df = pd.DataFrame(dd, index=[f"n={i}" for i in ns], columns=cols)
                st.dataframe(df)

                st.subheader(f"Polinomio de Newton (grado ≤ {orden_fin-1})")
                st.latex(P)
                st.subheader("Polinomio simplificado:")
                st.latex(P_simpl)

                st.subheader("Recurrencia básica:")
                st.latex(sp.Eq(sp.Function('f')(n) - sp.Function('f')(n-1), g_sym))

# --- tab7: Interpolación de Lagrange ---
with tab7:
    st.header("Interpolación de Lagrange y Error")

    # Entradas: función, nodos y punto de evaluación
    expr_lagrange = st.text_input("Ingrese la función f(x):", value="ln(1+x)", key="lagrange_fx")
    x_nodes = st.text_input("Ingrese los nodos x separados por comas:", value="0,0.6,0.9", key="lagrange_nodes")
    x_eval = st.number_input("Valor de x para evaluar el polinomio:", value=0.46, key="lagrange_eval")

    if st.button("Calcular Polinomio de Lagrange", key="lagrange_btn"):
        try:
            import sympy as sp
            # Definir símbolo y parsear función
            x = sp.Symbol('x')
            f_expr = sp.sympify(expr_lagrange, locals={'e': sp.E, 'ln': sp.log})
            f = sp.lambdify(x, f_expr, modules=['numpy'])

            # Convertir nodos a lista de floats y evaluar f en ellos
            x_list = [float(val) for val in x_nodes.split(",")]
            y_list = [f(val) for val in x_list]

            # Construir y mostrar cada base L_i(x)
            st.subheader("Términos L_i(x):")
            L_terms = []
            n = len(x_list)
            for i in range(n):
                Li = 1
                for j in range(n):
                    if i != j:
                        # (x - x_j)/(x_i - x_j)
                        Li *= (x - x_list[j])/(x_list[i] - x_list[j])
                Li_s = sp.simplify(Li)
                L_terms.append(Li_s)
                st.latex(f"L_{i}(x) = {sp.latex(Li_s)}")

            # Calcular y mostrar términos y_i * L_i(x)
            st.subheader("Términos y_i · L_i(x):")
            product_terms = []
            for i in range(n):
                term = sp.simplify(y_list[i] * L_terms[i])
                product_terms.append(term)
                st.latex(
                    f"y_{i} · L_{i}(x) = {sp.latex(y_list[i])}·{sp.latex(L_terms[i])} = {sp.latex(term)}"
                )

            # Polinomio completo P(x)
            P = sum(product_terms)
            P_expanded = sp.expand(P)
            st.subheader("Polinomio Interpolante P(x):")
            st.latex(f"P(x) = {sp.latex(P_expanded)}")

            # Evaluación en x_eval y cálculo del error
            p_func = sp.lambdify(x, P_expanded, modules=['numpy'])
            p_val = p_func(x_eval)
            f_val = f(x_eval)
            error = abs(f_val - p_val)

            st.subheader("Evaluación y Error:")
            st.write(f"f({x_eval}) = {f_val}")
            st.write(f"P({x_eval}) = {p_val}")
            st.write(f"Error absoluto = {error}")

        except Exception as e:
            st.error(f"Error al procesar la entrada: {e}")

# --- tab8: integra el sistema para aproximar un x con una precision de 10^n ---
with tab8:  
    st.header("Polinomio de Taylor con Aproximación 10⁻ⁿ")
    # funcion donde se realiza la logica para la aproximacion; recibe la funcion, un punto x0, y la presicion deseada. 
    def taylor_accuracy(f_expr, x, a, accuracy):
        # declara los valores
        f = sp.sympify(f_expr, locals={'e': sp.E, 'exp': sp.exp})
        T = f.subs(x, a)  # Primer término
        df = f
        n = 0
        estimate = float('inf')
        
        # garantiza que el error sea siempre menor que el n estimado
        while estimate > accuracy:
            n += 1
            df = sp.diff(df, x) # utilizado para conseguir la derivada n-esima
            term = df.subs(x, a) * (x - a)**n / sp.factorial(n) 
            T += term
            df_n1 = sp.diff(df, x)
            estimate = abs(df_n1.subs(x, a + 0.1) * (0.1)**(n+1)) / sp.factorial(n+1) #estima el siguiente termino con la derivada n+1
        
        return T, n #devuelve el polinomio completo
    
    # Aqui se registran las entradas del usuario
    a = st.number_input("Digite el punto al rededor del cual desea el polinomio (x0):", value=1.0, key="taylor_a2") # recibe un valor de x0 alrededor del cual se arma el polinomio de taylor
    accuracy = st.number_input("Digite la presicion deseada para el polinomio (valor de n en 10⁻ⁿ):", min_value=1, step=1, value=6, key="taylor_error") # solicita una presicion con la cual se desea el polinomio 
    expr = st.text_input("Función f(x):", value="x*exp(x)", key="taylor_fx1") # recibe una funcion f(x)

    if st.button("Calcular Polinomio"):
        x = sp.symbols('x')
        try:
            #declara el simbolo y parsea la función
            f_sym = sp.sympify(expr, locals={'e': sp.E, 'exp': sp.exp})
            epsilon = 10**(-accuracy) # declara la presicion que se desea que tenga el polinomio de taylor, en este caso digita unicamente el valor de n, n es siempre negativo
            T, n = taylor_accuracy(expr, x, a, epsilon)

            st.success(f"Polinomio de Taylor de grado {n} con error < {epsilon:.1e}") 
            st.latex(f"T_{n}(x) = {sp.latex(T.simplify())}")
            # necesario para graficar el polinomio de taylor, funciona igual al taylor incial.
            f_func = sp.lambdify(x, f_sym, modules=['numpy'])
            t_func = sp.lambdify(x, T, modules=['numpy'])
                
            x_vals = np.linspace(float(a) - 2, float(a) + 2, 400)
            y_vals_f = f_func(x_vals)
            y_vals_t = t_func(x_vals)
                
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals_f, label='f(x)', color='black')
            ax.plot(x_vals, y_vals_t, label=f'Taylor (n={n})', color='purple', linestyle='--')
            ax.axvline(a, color='red', linestyle=':', label=f'x = {a}')
            ax.set_title('Aproximación de Taylor')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al procesar la entrada: {e}") 

with tab9:
    st.header("Diferencias Divididas")

    st.markdown("### Ingrese los puntos (x, f(x))")
    puntos_input = st.text_area("Formato: x0,y0 | x1,y1 | x2,y2", value="1,1\n2,4\n3,9")
    x_eval = st.number_input("Valor de x a interpolar:", value=2.5, key="interp_x_eval")

    if st.button("Interpolar", key="interp_btn"):
        try:
            lines = puntos_input.strip().split("\n")
            data = [tuple(map(float, line.split(","))) for line in lines]
            xs, ys = zip(*data)

            # Tabla de diferencias divididas
            n = len(xs)
            dd = np.zeros((n, n))
            dd[:, 0] = ys
            for j in range(1, n):
                for i in range(n - j):
                    dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (xs[i+j] - xs[i])

            # Construcción del polinomio
            x = sp.Symbol('x')
            poly = 0
            for j in range(n):
                term = dd[0, j]
                for i in range(j):
                    term *= (x - xs[i])
                poly += term

            poly_simpl = sp.simplify(poly)
            st.latex(f"P(x) = {sp.latex(poly_simpl)}")

            # Evaluación
            p_func = sp.lambdify(x, poly_simpl, modules=['numpy'])
            y_interp = p_func(x_eval)
            st.write(f"P({x_eval}) = {y_interp}")

            # Tabla manejada por libreria pandas
            st.subheader("Tabla de diferencias divididas:")
            df_interp = pd.DataFrame(dd[:, :n], columns=[f"DD{j}" for j in range(n)])
            st.dataframe(df_interp)

            # Gráfico
            x_vals = np.linspace(min(xs) - 1, max(xs) + 1, 400)
            y_vals = p_func(x_vals)
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label="Interpolación", color="blue")
            ax.scatter(xs, ys, color='red', label="Puntos")
            ax.scatter(x_eval, y_interp, color='green', label=f"P({x_eval})")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al procesar la entrada: {e}")