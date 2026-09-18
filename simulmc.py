
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Simulación GBM - Monte Carlo",
    page_icon="📈",
    layout="wide"
)

st.title("Simulación Monte Carlo con Movimiento Geométrico Browniano")

st.write(
    """
    Esta aplicación utiliza el **Movimiento Geométrico Browniano (GBM)**
    para simular posibles precios futuros de diferentes activos financieros.
    """
)


# ============================================================
# UNIVERSO DE ACTIVOS
# ============================================================

TICKERS = [
"GAPB.MX","SPY", "GRUMAB.MX",  "NVDA","QUBT", "QTUM", "VOO", "QQQ", 
"AVGO", "ASURB.MX", "QBTS" , "^GSPC","BTC-USD", "BABA", "VISTAA.MX", "DANHOS13.MX", "EDUCA18.MX",
"FIBRAMQ12.MX", "FIBRAPL14.MX", "FIHO12.MX", "FINN13.MX", "FMTY14.MX",
"FPLUS16.MX", "FSHOP13.MX", "FUNO11.MX", "ACCELSAB.MX", "AGUA.MX", 
"CADUA.MX",  "DINEB.MX",  "GCARSOA1.MX",
"GISSAA.MX", "GMD.MX",  "HOMEX.MX",  "KUOB.MX",
"OMAB.MX", "ORBIA.MX", "PASAB.MX", "PINFRA.MX", "SITES1A-1.MX", "TMMA.MX",
"TRAXIONA.MX", "VESTA.MX", "VINTE.MX", "VOLARA.MX", "ALPEKA.MX", "AUTLANB.MX",
"CEMEXCPO.MX", "CMOCTEZ.MX", "COLLADO.MX", "CONVERA.MX", "CYDSASAA.MX",
"GCC.MX", "GMEXICOB.MX", "ICHB.MX", "LAMOSA.MX", "MFRISCOA-1.MX",
"PE&OLES.MX", "POCHTECB.MX", "SIMECB.MX", "TEAKCPO.MX", "VITROA.MX",
"AC.MX", "BIMBOA.MX", "CHDRAUIB.MX", "CUERVO.MX", "CULTIBAB.MX",
"FEMSAUBD.MX", "GIGANTE.MX", "GRUMAB.MX", "HERDEZ.MX", "KIMBERA.MX",
"KOFUBL.MX", "LACOMERUBC.MX", "MINSAB.MX", "SORIANAB.MX", "WALMEX.MX",
"BEVIDESB.MX", "FRAGUAB.MX", "LABB.MX", "MEDICAB.MX", "AMXB.MX",
"AXTELCPO.MX", "CABLECPO.MX", "CTAXTELA.MX", "MEGACPO.MX", "TLEVISACPO.MX",
"ACTINVRB.MX", "BBAJIOO.MX", "BOLSAA.MX", "CREAL.MX", "FINAMEXO.MX",
"FINDEP.MX", "GBMO.MX", "GENTERA.MX", "GFINBURO.MX", "GFNORTEO.MX",
"GNP.MX", "GPROFUT.MX", "INVEXA.MX", "PROCORPB.MX", "Q.MX", "RA.MX",
"AGUILASCPO.MX", "ALSEA.MX", "CIDMEGA.MX", "CIEB.MX", "CMRB.MX",
"HCITY.MX", "HOTEL.MX", "LIVEPOL1.MX", "NEMAKA.MX", "POSADASA.MX",
"RLHA.MX", "SPORTS.MX", "VASCONI.MX", "ARKB", "BTCW", "BTCO", "BITB",
"HODL", "EZBC", "FBTC", "BRRR", "GBTC", "DEFI", "IBIT", "ACWI",
"FAS", "SPXL", "TECL", "IAU", "NU", "MELI", "META",
"JPM","NFLX", "IONQ", "RGTI", "PLTR", "SOFI", "HOOD", "FRES.MX", "MCHI", "INDA", "TSM", "AMD", "GOOGL", "AMZN","V"

]


# ============================================================
# PARÁMETROS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    periodo = st.selectbox(
        "Periodo histórico",
        [
            "6 meses",
            "1 año",
            "2 años",
            "3 años",
            "5 años",
            "10 años"
        ],
        index=2
    )


with col2:

    simulaciones = st.number_input(
        "Número de simulaciones",
        min_value=100,
        max_value=50000,
        value=10000,
        step=1000
    )


periodos_yf = {
    "6 meses": "6mo",
    "1 año": "1y",
    "2 años": "2y",
    "3 años": "3y",
    "5 años": "5y",
    "10 años": "10y"
}

periodo_yf = periodos_yf[periodo]

INVERSION = 100000


# ============================================================
# EXPLICACIÓN DEL MODELO
# ============================================================

with st.expander("Modelo matemático"):

    st.latex(
        r"""
        dS_t=\mu S_t\,dt+\sigma S_t\,dW_t
        """
    )

    st.write(
        """
        La solución exacta del Movimiento Geométrico Browniano utilizada
        en la simulación es:
        """
    )

    st.latex(
        r"""
        S_{t+\Delta t}
        =
        S_t
        \exp
        \left[
        \left(\mu-\frac{1}{2}\sigma^2\right)\Delta t
        +
        \sigma\sqrt{\Delta t}Z_t
        \right]
        """
    )

    st.write(
        """
        donde:

        - $S_t$ = precio del activo
        - $\\mu$ = rendimiento medio
        - $\\sigma$ = volatilidad
        - $\\Delta t$ = intervalo temporal
        - $Z_t \\sim N(0,1)$
        """
    )


# ============================================================
# DESCARGA DE PRECIOS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def obtener_precios(ticker, periodo):

    try:

        df = yf.download(
            ticker,
            period=periodo,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            multi_level_index=False
        )

        if df is None or df.empty:
            return None

        if "Close" not in df.columns:
            return None

        precios = df["Close"].dropna()

        if len(precios) < 30:
            return None

        return precios

    except Exception:
        return None


# ============================================================
# SIMULACIÓN GBM TERMINAL
# ============================================================

def simular_terminal_gbm(
    S0,
    mu_anual,
    sigma_anual,
    dias,
    numero_simulaciones,
    semilla
):

    T = dias / 252

    rng = np.random.default_rng(semilla)

    Z = rng.normal(
        0,
        1,
        int(numero_simulaciones)
    )

    log_ST = (
        np.log(S0)
        +
        (mu_anual - 0.5 * sigma_anual**2) * T
        +
        sigma_anual * np.sqrt(T) * Z
    )

    precio_final = np.exp(log_ST)

    return precio_final


# ============================================================
# TRAYECTORIAS PARA LOS 5 MEJORES
# ============================================================

def generar_trayectorias(
    S0,
    mu_anual,
    sigma_anual,
    dias,
    numero_trayectorias=100,
    semilla=42
):

    dt = 1 / 252

    rng = np.random.default_rng(semilla)

    Z = rng.normal(
        0,
        1,
        (numero_trayectorias, dias)
    )

    incremento = (
        (mu_anual - 0.5 * sigma_anual**2) * dt
        +
        sigma_anual * np.sqrt(dt) * Z
    )

    trayectorias = (
        S0
        *
        np.exp(
            np.cumsum(incremento, axis=1)
        )
    )

    return trayectorias


# ============================================================
# BOTÓN
# ============================================================

if st.button(
    "Ejecutar simulación",
    type="primary",
    use_container_width=True
):

    resultados = []

    progreso = st.progress(0)

    mensaje = st.empty()

    total_activos = len(TICKERS)


    # ========================================================
    # PROCESAMIENTO DE ACTIVOS
    # ========================================================

    for i, ticker in enumerate(TICKERS):

        mensaje.text(
            f"Procesando {ticker} "
            f"({i + 1}/{total_activos})"
        )

        precios = obtener_precios(
            ticker,
            periodo_yf
        )

        if precios is None:
            progreso.progress(
                (i + 1) / total_activos
            )
            continue


        # ----------------------------------------------------
        # RENDIMIENTOS LOGARÍTMICOS
        # ----------------------------------------------------

        log_returns = np.log(
            precios / precios.shift(1)
        ).dropna()

        if len(log_returns) < 30:
            continue


        # ----------------------------------------------------
        # PARÁMETROS GBM
        # ----------------------------------------------------

        mu_diaria = log_returns.mean()

        sigma_diaria = log_returns.std()

        mu_anual = mu_diaria * 252

        sigma_anual = sigma_diaria * np.sqrt(252)


        # ----------------------------------------------------
        # PRECIO INICIAL
        # ----------------------------------------------------

        S0 = float(
            precios.iloc[-1]
        )


        # ----------------------------------------------------
        # HORIZONTE
        #
        # Mismo número de días que el periodo histórico
        # ----------------------------------------------------

        dias = len(log_returns)


        # ----------------------------------------------------
        # SIMULACIÓN TERMINAL
        # ----------------------------------------------------

        precio_final = simular_terminal_gbm(
            S0=S0,
            mu_anual=mu_anual,
            sigma_anual=sigma_anual,
            dias=dias,
            numero_simulaciones=simulaciones,
            semilla=1000 + i
        )


        # ----------------------------------------------------
        # RENDIMIENTO
        # ----------------------------------------------------

        rendimiento = (
            precio_final / S0
        ) - 1


        # ----------------------------------------------------
        # VALOR DE LA INVERSIÓN
        # ----------------------------------------------------

        valor_final = (
            INVERSION
            *
            (1 + rendimiento)
        )


        ganancia = (
            valor_final
            -
            INVERSION
        )


        # ----------------------------------------------------
        # ESTADÍSTICAS
        # ----------------------------------------------------

        rendimiento_promedio = np.mean(
            rendimiento
        )

        mediana = np.median(
            rendimiento
        )

        p5 = np.percentile(
            rendimiento,
            5
        )

        p95 = np.percentile(
            rendimiento,
            95
        )

        valor_promedio = np.mean(
            valor_final
        )

        ganancia_promedio = np.mean(
            ganancia
        )


        # ----------------------------------------------------
        # VaR 95%
        # ----------------------------------------------------

        VaR_95 = max(
            0,
            -INVERSION * p5
        )


        # ----------------------------------------------------
        # CVaR 95%
        # ----------------------------------------------------

        cola = rendimiento[
            rendimiento <= p5
        ]

        if len(cola) > 0:

            cvar_rendimiento = np.mean(
                cola
            )

        else:

            cvar_rendimiento = p5


        CVaR_95 = max(
            0,
            -INVERSION * cvar_rendimiento
        )


        # ----------------------------------------------------
        # PROBABILIDAD DE GANANCIA
        # ----------------------------------------------------

        prob_ganancia = np.mean(
            rendimiento > 0
        )


        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        resultados.append({

            "Ticker": ticker,

            "Rendimiento promedio":
                rendimiento_promedio,

            "Mediana":
                mediana,

            "Valor final promedio":
                valor_promedio,

            "Ganancia promedio":
                ganancia_promedio,

            "VaR 95%":
                VaR_95,

            "CVaR 95%":
                CVaR_95,

            "P5":
                p5,

            "P95":
                p95,

            "Probabilidad de ganancia":
                prob_ganancia,

            "μ anual":
                mu_anual,

            "σ anual":
                sigma_anual,

            "Precio actual":
                S0,

            "Días simulados":
                dias
        })


        progreso.progress(
            (i + 1) / total_activos
        )


    mensaje.empty()
    progreso.empty()


    # ========================================================
    # VALIDACIÓN
    # ========================================================

    if len(resultados) == 0:

        st.error(
            "No fue posible obtener datos suficientes "
            "para ninguno de los activos."
        )

        st.stop()


    # ========================================================
    # DATAFRAME
    # ========================================================

    df_resultados = pd.DataFrame(
        resultados
    )


    # ========================================================
    # ORDENAMIENTO
    # ========================================================

    df_resultados = (
        df_resultados
        .sort_values(
            "Rendimiento promedio",
            ascending=False
        )
        .reset_index(drop=True)
    )


    df_resultados.insert(
        0,
        "Ranking",
        np.arange(
            1,
            len(df_resultados) + 1
        )
    )


    # ========================================================
    # TOP 5
    # ========================================================

    top5 = df_resultados.head(5)


    # ========================================================
    # RESULTADO PRINCIPAL
    # ========================================================

    st.success(
        f"Simulación terminada. "
        f"Se analizaron {len(df_resultados)} activos."
    )


    st.subheader(
        "Ranking de activos"
    )


    # ========================================================
    # TABLA
    # ========================================================

    tabla = df_resultados.copy()

    tabla["Rendimiento promedio"] = (
        tabla["Rendimiento promedio"]
        .map(lambda x: f"{x:.2%}")
    )

    tabla["Mediana"] = (
        tabla["Mediana"]
        .map(lambda x: f"{x:.2%}")
    )

    tabla["Valor final promedio"] = (
        tabla["Valor final promedio"]
        .map(lambda x: f"${x:,.2f}")
    )

    tabla["Ganancia promedio"] = (
        tabla["Ganancia promedio"]
        .map(lambda x: f"${x:,.2f}")
    )

    tabla["VaR 95%"] = (
        tabla["VaR 95%"]
        .map(lambda x: f"${x:,.2f}")
    )

    tabla["CVaR 95%"] = (
        tabla["CVaR 95%"]
        .map(lambda x: f"${x:,.2f}")
    )

    tabla["P5"] = (
        tabla["P5"]
        .map(lambda x: f"{x:.2%}")
    )

    tabla["P95"] = (
        tabla["P95"]
        .map(lambda x: f"{x:.2%}")
    )

    tabla["Probabilidad de ganancia"] = (
        tabla["Probabilidad de ganancia"]
        .map(lambda x: f"{x:.2%}")
    )

    tabla["μ anual"] = (
        tabla["μ anual"]
        .map(lambda x: f"{x:.2%}")
    )

    tabla["σ anual"] = (
        tabla["σ anual"]
        .map(lambda x: f"{x:.2%}")
    )

    tabla["Precio actual"] = (
        tabla["Precio actual"]
        .map(lambda x: f"${x:,.2f}")
    )


    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # TOP 1
    # ========================================================

    ganador = df_resultados.iloc[0]


    st.subheader(
        "Activo con mayor rendimiento promedio simulado"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Activo",
            ganador["Ticker"]
        )


    with c2:

        st.metric(
            "Rendimiento promedio",
            f"{ganador['Rendimiento promedio']:.2%}"
        )


    with c3:

        st.metric(
            "Valor final promedio",
            f"${ganador['Valor final promedio']:,.2f}"
        )


    with c4:

        st.metric(
            "Probabilidad de ganancia",
            f"{ganador['Probabilidad de ganancia']:.2%}"
        )


    # ========================================================
    # GRÁFICA DE RENDIMIENTOS
    # ========================================================

    st.subheader(
        "Rendimiento promedio simulado"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        df_resultados["Ticker"],
        df_resultados["Rendimiento promedio"]
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_ylabel(
        "Rendimiento"
    )

    ax.set_xlabel(
        "Activo"
    )

    ax.set_title(
        "Ranking por rendimiento promedio simulado"
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # VALOR FINAL DE LA INVERSIÓN
    # ========================================================

    st.subheader(
        "Valor final promedio de una inversión de $100,000"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        df_resultados["Ticker"],
        df_resultados["Valor final promedio"]
    )

    ax.axhline(
        INVERSION,
        linestyle="--",
        linewidth=1,
        label="$100,000 iniciales"
    )

    ax.set_ylabel(
        "Valor final ($)"
    )

    ax.set_xlabel(
        "Activo"
    )

    ax.set_title(
        "Valor final promedio simulado"
    )

    ax.legend()

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # VaR Y CVaR
    # ========================================================

    st.subheader(
        "Riesgo: VaR y CVaR al 95%"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x = np.arange(
        len(df_resultados)
    )

    ancho = 0.35

    ax.bar(
        x - ancho / 2,
        df_resultados["VaR 95%"],
        width=ancho,
        label="VaR 95%"
    )

    ax.bar(
        x + ancho / 2,
        df_resultados["CVaR 95%"],
        width=ancho,
        label="CVaR 95%"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        df_resultados["Ticker"],
        rotation=90
    )

    ax.set_ylabel(
        "Pérdida potencial ($)"
    )

    ax.set_xlabel(
        "Activo"
    )

    ax.set_title(
        "VaR y CVaR del rendimiento acumulado al horizonte simulado"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # TRAYECTORIAS TOP 5
    # ========================================================

    st.subheader(
        "Trayectorias simuladas de los 5 activos con mayor rendimiento"
    )


    for posicion, (_, fila) in enumerate(
        top5.iterrows(),
        start=1
    ):

        ticker = fila["Ticker"]

        precios = obtener_precios(
            ticker,
            periodo_yf
        )

        if precios is None:
            continue


        log_returns = np.log(
            precios / precios.shift(1)
        ).dropna()


        mu_diaria = log_returns.mean()

        sigma_diaria = log_returns.std()

        mu_anual = mu_diaria * 252

        sigma_anual = sigma_diaria * np.sqrt(252)

        S0 = float(
            precios.iloc[-1]
        )

        dias = len(log_returns)


        trayectorias = generar_trayectorias(
            S0=S0,
            mu_anual=mu_anual,
            sigma_anual=sigma_anual,
            dias=dias,
            numero_trayectorias=100,
            semilla=5000 + posicion
        )


        # ----------------------------------------------------
        # GRÁFICA
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )


        for trayectoria in trayectorias:

            ax.plot(
                trayectoria,
                alpha=0.08
            )


        # Media de las trayectorias

        media_trayectorias = np.mean(
            trayectorias,
            axis=0
        )

        ax.plot(
            media_trayectorias,
            linewidth=2,
            label="Media simulada"
        )


        ax.axhline(
            S0,
            linestyle="--",
            linewidth=1,
            label="Precio actual"
        )


        ax.set_title(
            f"{posicion}. {ticker} — Trayectorias GBM"
        )

        ax.set_xlabel(
            "Días de simulación"
        )

        ax.set_ylabel(
            "Precio"
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # ========================================================
    # DESCARGA CSV
    # ========================================================

    csv = df_resultados.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="Descargar resultados en CSV",
        data=csv,
        file_name="resultados_gbm.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # INTERPRETACIÓN
    # ========================================================

    st.info(
        """
        **Interpretación:**

        El ranking se construye a partir del rendimiento promedio
        de los precios terminales simulados mediante GBM.

        El VaR 95% representa la pérdida potencial asociada al percentil
        5% de los rendimientos simulados, considerando una inversión
        inicial de $100,000.

        El CVaR 95% representa la pérdida promedio dentro de la cola
        de peores resultados, también para una inversión inicial de
        $100,000.

        El horizonte de simulación utiliza el mismo número de días
        de negociación que el periodo histórico seleccionado.
        """
    )
