import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Simulador de Destilación Binaria",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚗️ Simulador de Destilación Binaria")

st.caption(
    "McCabe-Thiele + Flash Rachford-Rice para mezclas binarias "
    "ideales. Se utiliza la Ley de Raoult."
)


# ============================================================
# 2. BASE DE DATOS TERMODINÁMICA
# ============================================================
#
# Antoine:
#
# log10(Psat [bar]) = A - B / (T [°C] + C)
#
# Los coeficientes corresponden a la forma de Antoine
# expresada con T en °C y presión en bar.
#
# IMPORTANTE:
# Los rangos de validez son aproximados y se utilizan para
# lanzar advertencias, no como límites matemáticos estrictos.
# ============================================================

COMPONENTS = {

    "Benceno": {
        "antoine": (4.01814, 1203.835, 219.800),
        "T_range": (14.0, 81.0),
    },

    "Tolueno": {
        "antoine": (4.07827, 1343.943, 219.377),
        "T_range": (30.0, 111.0),
    },

    "o-Xileno": {
        "antoine": (4.12078, 1474.679, 213.686),
        "T_range": (20.0, 145.0),
    },

    "m-Xileno": {
        "antoine": (4.13377, 1462.242, 215.105),
        "T_range": (20.0, 145.0),
    },

    "p-Xileno": {
        "antoine": (4.11432, 1453.430, 215.307),
        "T_range": (20.0, 145.0),
    },

    "n-Pentano": {
        "antoine": (3.97786, 1064.840, 232.014),
        "T_range": (-20.0, 36.0),
    },

    "n-Hexano": {
        "antoine": (4.00266, 1171.530, 224.216),
        "T_range": (0.0, 69.0),
    },

    "n-Heptano": {
        "antoine": (4.02832, 1268.636, 216.900),
        "T_range": (20.0, 98.0),
    },

    "n-Octano": {
        "antoine": (4.04867, 1355.126, 209.517),
        "T_range": (30.0, 126.0),
    },

    "Metanol": {
        "antoine": (5.20409, 1581.341, 239.500),
        "T_range": (-20.0, 65.0),
    },

    "Etanol": {
        "antoine": (5.24677, 1598.673, 226.574),
        "T_range": (0.0, 78.0),
    },

    "Agua": {
        "antoine": (5.11564, 1687.537, 230.170),
        "T_range": (1.0, 100.0),
    },
}


# ============================================================
# 3. FUNCIONES TERMODINÁMICAS
# ============================================================

def psat(T, antoine):
    """
    Presión de vapor en bar.

    Antoine:
    log10(Psat) = A - B/(T+C)

    T en °C.
    """

    A, B, C = antoine

    denominator = T + C

    if np.any(np.asarray(denominator) <= 0):
        raise ValueError("Temperatura incompatible con la ecuación de Antoine.")

    return 10.0 ** (A - B / denominator)


def tsat(P, antoine):
    """
    Temperatura de saturación en °C.
    """

    if P <= 0:
        raise ValueError("La presión debe ser positiva.")

    A, B, C = antoine

    denominator = A - np.log10(P)

    if abs(denominator) < 1e-12:
        raise ValueError("No se puede calcular la temperatura de saturación.")

    return B / denominator - C


def K_values(T, P, ant_lk, ant_hk):
    """
    Coeficientes de equilibrio K_i = Psat_i / P
    para una mezcla ideal.
    """

    if P <= 0:
        raise ValueError("La presión debe ser positiva.")

    return np.array([
        psat(T, ant_lk) / P,
        psat(T, ant_hk) / P,
    ])


def bubble_pressure(T, x_lk, ant_lk, ant_hk):
    """
    Presión de burbuja para una mezcla binaria ideal.
    """

    x_lk = np.clip(x_lk, 0.0, 1.0)

    return (
        x_lk * psat(T, ant_lk)
        + (1.0 - x_lk) * psat(T, ant_hk)
    )


def check_antoine_range(T, component):
    """
    Comprueba si T está dentro del rango aproximado
    de los coeficientes de Antoine.
    """

    Tmin, Tmax = COMPONENTS[component]["T_range"]

    return Tmin <= T <= Tmax


# ============================================================
# 4. CURVA DE EQUILIBRIO
# ============================================================

@st.cache_data
def get_xy_equilibrium_curve(P, ant_lk, ant_hk, points=501):

    if P <= 0:
        raise ValueError("La presión debe ser positiva.")

    x_values = np.linspace(0.0, 1.0, points)
    y_values = np.zeros_like(x_values)
    T_values = np.zeros_like(x_values)

    T_lk = tsat(P, ant_lk)
    T_hk = tsat(P, ant_hk)

    T_low = min(T_lk, T_hk)
    T_high = max(T_lk, T_hk)

    for i, x in enumerate(x_values):

        if x <= 0.0:
            T_values[i] = T_hk
            y_values[i] = 0.0
            continue

        if x >= 1.0:
            T_values[i] = T_lk
            y_values[i] = 1.0
            continue

        def bubble_residual(T):
            return (
                x * psat(T, ant_lk)
                + (1.0 - x) * psat(T, ant_hk)
                - P
            )

        try:
            T = brentq(
                bubble_residual,
                T_low,
                T_high,
            )
        except ValueError as exc:
            raise ValueError(
                f"No se pudo calcular el equilibrio para x={x:.4f}."
            ) from exc

        T_values[i] = T

        y_values[i] = (
            x * psat(T, ant_lk) / P
        )

    y_values = np.clip(y_values, 0.0, 1.0)

    if np.any(np.diff(y_values) <= 0):
        raise ValueError(
            "La curva de equilibrio no es estrictamente creciente. "
            "Revisa los datos termodinámicos."
        )

    return x_values, y_values, T_values


def make_equilibrium_functions(x_eq, y_eq):

    y_interp = PchipInterpolator(
        x_eq,
        y_eq,
        extrapolate=False
    )

    x_interp = PchipInterpolator(
        y_eq,
        x_eq,
        extrapolate=False
    )

    def y_from_x(x):
        x = float(np.clip(x, 0.0, 1.0))
        return float(y_interp(x))

    def x_from_y(y):
        y = float(np.clip(y, 0.0, 1.0))
        return float(x_interp(y))

    return y_from_x, x_from_y


# ============================================================
# 5. MCCABE-THIELE
# ============================================================

def q_line(x, zf, q):

    if abs(q - 1.0) < 1e-10:
        raise ValueError(
            "Para q = 1 la línea q es vertical."
        )

    return (
        q / (q - 1.0) * x
        - zf / (q - 1.0)
    )


def find_pinch(
    zf,
    xb,
    xd,
    q,
    y_from_x,
):
    """
    Busca el punto de intersección entre la línea q
    y la curva de equilibrio dentro de la zona relevante.
    """

    if abs(q - 1.0) < 1e-10:

        yp = y_from_x(zf)

        return zf, yp

    def residual(x):

        return q_line(x, zf, q) - y_from_x(x)

    grid = np.linspace(
        xb,
        xd,
        2001
    )

    values = np.array(
        [residual(x) for x in grid]
    )

    roots = []

    for i in range(len(grid) - 1):

        f1 = values[i]
        f2 = values[i + 1]

        if abs(f1) < 1e-10:
            roots.append(grid[i])

        elif f1 * f2 < 0:

            root = brentq(
                residual,
                grid[i],
                grid[i + 1]
            )

            roots.append(root)

    if not roots:

        raise ValueError(
            "No se encontró intersección entre la línea q "
            "y la curva de equilibrio."
        )

    # Elegimos el punto más próximo a zF.
    xp = min(
        roots,
        key=lambda x: abs(x - zf)
    )

    yp = y_from_x(xp)

    return xp, yp


def minimum_reflux(xd, xp, yp):

    denominator = yp - xp

    if denominator <= 1e-12:

        raise ValueError(
            "No se puede calcular Rmin: "
            "el pinch está demasiado cerca de la diagonal."
        )

    Rmin = (xd - yp) / denominator

    if Rmin <= 0:

        raise ValueError(
            "El reflujo mínimo calculado no es físicamente válido."
        )

    return Rmin


def operating_lines(
    xd,
    xb,
    x_intersection,
    y_intersection,
    R
):

    # Línea de rectificación
    m_rect = R / (R + 1.0)
    b_rect = xd / (R + 1.0)

    # Línea de agotamiento
    if abs(x_intersection - xb) < 1e-12:

        raise ValueError(
            "La intersección de las líneas está demasiado cerca de xB."
        )

    m_strip = (
        y_intersection - xb
    ) / (
        x_intersection - xb
    )

    b_strip = xb * (1.0 - m_strip)

    return (
        m_rect,
        b_rect,
        m_strip,
        b_strip
    )


def calculate_intersection(
    zf,
    q,
    m_rect,
    b_rect
):

    if abs(q - 1.0) < 1e-10:

        x_int = zf
        y_int = (
            m_rect * x_int
            + b_rect
        )

        return x_int, y_int

    m_q = q / (q - 1.0)
    b_q = -zf / (q - 1.0)

    denominator = m_rect - m_q

    if abs(denominator) < 1e-12:

        raise ValueError(
            "La línea q y la línea de rectificación "
            "son prácticamente paralelas."
        )

    x_int = (
        b_q - b_rect
    ) / denominator

    y_int = (
        m_rect * x_int
        + b_rect
    )

    return x_int, y_int


def calculate_stages(
    xd,
    xb,
    x_int,
    y_int,
    m_rect,
    b_rect,
    m_strip,
    b_strip,
    x_from_y,
    max_stages=200
):

    if not xb < x_int < xd:

        raise ValueError(
            "La intersección de las líneas de operación "
            "está fuera de la región válida."
        )

    stage_x = [xd]
    stage_y = [xd]

    rows = []

    current_y = xd

    reached_bottoms = False
    warning_limit = False

    previous_x = None

    for stage in range(1, max_stages + 1):

        # Movimiento horizontal hacia la curva de equilibrio
        x_liquid = x_from_y(current_y)

        stage_x.append(x_liquid)
        stage_y.append(current_y)

        # Determinación de la sección
        if x_liquid >= x_int:
            section = "Rectificación"
            y_operating = (
                m_rect * x_liquid
                + b_rect
            )
        else:
            section = "Agotamiento"
            y_operating = (
                m_strip * x_liquid
                + b_strip
            )

        # ----------------------------------------------------
        # Comprobación de llegada a fondos
        # ----------------------------------------------------

        if x_liquid <= xb:

            reached_bottoms = True

            if previous_x is not None:

                denominator = (
                    previous_x - x_liquid
                )

                if abs(denominator) > 1e-12:

                    fraction = (
                        previous_x - xb
                    ) / denominator

                    fraction = float(
                        np.clip(
                            fraction,
                            0.0,
                            1.0
                        )
                    )

                else:
                    fraction = 1.0

            else:

                fraction = 1.0

        else:

            fraction = 1.0

        rows.append({
            "Etapa": stage,
            "x_líquido": round(x_liquid, 5),
            "y_vapor": round(current_y, 5),
            "Fracción de etapa": round(
                fraction,
                4
            ),
            "Sección": section
        })

        # ----------------------------------------------------
        # Finalización
        # ----------------------------------------------------

        if x_liquid <= xb:

            break

        previous_x = x_liquid

        # Movimiento vertical
        stage_x.append(x_liquid)
        stage_y.append(y_operating)

        current_y = y_operating

    else:

        warning_limit = True

    df = pd.DataFrame(rows)

    if reached_bottoms:

        n_full = len(rows) - 1

        final_fraction = rows[-1][
            "Fracción de etapa"
        ]

        equivalent_stages = (
            n_full
            + final_fraction
        )

    else:

        equivalent_stages = float(
            len(rows)
        )

    return (
        stage_x,
        stage_y,
        df,
        len(rows),
        equivalent_stages,
        reached_bottoms,
        warning_limit
    )


# ============================================================
# 6. RACHFORD-RICE
# ============================================================

def rachford_rice(psi, z, K):

    denominator = (
        1.0
        + psi * (K - 1.0)
    )

    if np.any(
        np.abs(denominator) < 1e-12
    ):

        raise ValueError(
            "Denominador cercano a cero en Rachford-Rice."
        )

    return np.sum(
        z * (K - 1.0)
        / denominator
    )


def flash_z_P_psi(
    z_lk,
    P,
    psi,
    ant_lk,
    ant_hk
):

    if not 0.0 < z_lk < 1.0:
        raise ValueError(
            "La composición de alimentación debe estar entre 0 y 1."
        )

    if P <= 0:
        raise ValueError(
            "La presión debe ser positiva."
        )

    if not 0.0 < psi < 1.0:
        raise ValueError(
            "La fracción vaporizada debe estar entre 0 y 1."
        )

    z = np.array([
        z_lk,
        1.0 - z_lk
    ])

    T_lk = tsat(P, ant_lk)
    T_hk = tsat(P, ant_hk)

    T_low = min(T_lk, T_hk)
    T_high = max(T_lk, T_hk)

    def residual(T):

        K = K_values(
            T,
            P,
            ant_lk,
            ant_hk
        )

        return rachford_rice(
            psi,
            z,
            K
        )

    try:

        T = brentq(
            residual,
            T_low,
            T_high
        )

    except ValueError as exc:

        raise ValueError(
            "No existe una solución bifásica para "
            "la presión, composición y fracción "
            "vaporizada seleccionadas."
        ) from exc

    K = K_values(
        T,
        P,
        ant_lk,
        ant_hk
    )

    denominator = (
        1.0
        + psi * (K - 1.0)
    )

    x = z / denominator
    y = K * x

    # Normalización por seguridad numérica
    x /= np.sum(x)
    y /= np.sum(y)

    return T, x, y


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.header("🧪 Selección de mezcla")

comp_list = list(COMPONENTS.keys())

col_a, col_b = st.sidebar.columns(2)

comp_a = col_a.selectbox(
    "Componente A",
    comp_list,
    index=comp_list.index("Benceno")
)

comp_b = col_b.selectbox(
    "Componente B",
    comp_list,
    index=comp_list.index("Tolueno")
)

if comp_a == comp_b:

    st.error(
        "Selecciona dos componentes diferentes."
    )

    st.stop()


# ============================================================
# 8. PRESIÓN
# ============================================================

st.sidebar.header(
    "⚙️ Parámetros de operación"
)

P_col = st.sidebar.number_input(
    "Presión absoluta (bar)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.1
)


# ============================================================
# 9. IDENTIFICACIÓN AUTOMÁTICA LK / HK
# ============================================================

T_a = tsat(
    P_col,
    COMPONENTS[comp_a]["antoine"]
)

T_b = tsat(
    P_col,
    COMPONENTS[comp_b]["antoine"]
)

if T_a < T_b:

    lk = comp_a
    hk = comp_b

else:

    lk = comp_b
    hk = comp_a


ant_lk = COMPONENTS[lk]["antoine"]
ant_hk = COMPONENTS[hk]["antoine"]


st.sidebar.success(
    f"🟢 Light Key: **{lk}**\n\n"
    f"🔴 Heavy Key: **{hk}**"
)


# ============================================================
# 10. PARÁMETROS MCCABE-THIELE
# ============================================================

zf = st.sidebar.slider(
    f"zF ({lk})",
    0.05,
    0.90,
    0.30,
    0.01
)

xd_min = min(
    0.99,
    zf + 0.01
)

xd = st.sidebar.slider(
    f"xD ({lk})",
    xd_min,
    0.99,
    max(
        xd_min,
        0.95
    ),
    0.01
)

xb_max = max(
    0.01,
    zf - 0.01
)

xb_default = min(
    0.07,
    xb_max
)

xb = st.sidebar.slider(
    f"xB ({lk})",
    0.01,
    xb_max,
    xb_default,
    0.01
)

frac_vap = st.sidebar.slider(
    "Fracción vaporizada de alimentación ψ",
    0.0,
    1.0,
    0.40,
    0.01
)

q_val = 1.0 - frac_vap

factor_R = st.sidebar.slider(
    "R / Rmin",
    1.01,
    4.00,
    1.50,
    0.05
)


# ============================================================
# 11. TABS
# ============================================================

tab_mt, tab_flash = st.tabs(
    [
        "📊 McCabe-Thiele",
        "⚡ Flash Rachford-Rice"
    ]
)


# ============================================================
# 12. MCCABE-THIELE
# ============================================================

with tab_mt:

    try:

        # ----------------------------------------------------
        # Validaciones
        # ----------------------------------------------------

        if not (
            0.0 < xb < zf < xd < 1.0
        ):

            raise ValueError(
                "Debe cumplirse: 0 < xB < zF < xD < 1."
            )

        # ----------------------------------------------------
        # Curva de equilibrio
        # ----------------------------------------------------

        (
            x_eq,
            y_eq,
            T_eq
        ) = get_xy_equilibrium_curve(
            P_col,
            ant_lk,
            ant_hk
        )

        (
            y_from_x,
            x_from_y
        ) = make_equilibrium_functions(
            x_eq,
            y_eq
        )

        # ----------------------------------------------------
        # Pinch y reflujo mínimo
        # ----------------------------------------------------

        xp, yp = find_pinch(
            zf,
            xb,
            xd,
            q_val,
            y_from_x
        )

        Rmin = minimum_reflux(
            xd,
            xp,
            yp
        )

        R = factor_R * Rmin

        # ----------------------------------------------------
        # Línea de rectificación
        # ----------------------------------------------------

        m_rect = R / (R + 1.0)
        b_rect = xd / (R + 1.0)

        # ----------------------------------------------------
        # Intersección con línea q
        # ----------------------------------------------------

        x_int, y_int = calculate_intersection(
            zf,
            q_val,
            m_rect,
            b_rect
        )

        if not (
            xb < x_int < xd
        ):

            raise ValueError(
                "La intersección entre las líneas de "
                "operación está fuera del rango "
                "de composición permitido."
            )

        # ----------------------------------------------------
        # Líneas de operación
        # ----------------------------------------------------

        (
            m_rect,
            b_rect,
            m_strip,
            b_strip
        ) = operating_lines(
            xd,
            xb,
            x_int,
            y_int,
            R
        )

        # ----------------------------------------------------
        # Etapas
        # ----------------------------------------------------

        (
            stage_x,
            stage_y,
            df_stages,
            n_stages,
            n_equiv,
            reached_bottoms,
            warning_limit
        ) = calculate_stages(
            xd,
            xb,
            x_int,
            y_int,
            m_rect,
            b_rect,
            m_strip,
            b_strip,
            x_from_y
        )

        if warning_limit:

            st.warning(
                "Se alcanzó el límite de 200 etapas. "
                "El sistema está muy cerca de Rmin."
            )

        if not reached_bottoms:

            st.warning(
                "No se alcanzó la composición de fondos "
                "dentro de las 200 etapas."
            )

        # ----------------------------------------------------
        # Temperaturas
        # ----------------------------------------------------

        T_bubble_pure_lk = tsat(
            P_col,
            ant_lk
        )

        T_bubble_pure_hk = tsat(
            P_col,
            ant_hk
        )

        # ----------------------------------------------------
        # Avisos Antoine
        # ----------------------------------------------------

        if not check_antoine_range(
            T_bubble_pure_lk,
            lk
        ):

            st.warning(
                f"La temperatura de saturación del "
                f"{lk} está fuera del rango aproximado "
                f"de los datos de Antoine."
            )

        if not check_antoine_range(
            T_bubble_pure_hk,
            hk
        ):

            st.warning(
                f"La temperatura de saturación del "
                f"{hk} está fuera del rango aproximado "
                f"de los datos de Antoine."
            )

        # ====================================================
        # MÉTRICAS
        # ====================================================

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rmin",
            f"{Rmin:.3f}"
        )

        c2.metric(
            "R operación",
            f"{R:.3f}",
            f"{factor_R:.2f} × Rmin"
        )

        c3.metric(
            "Etapas",
            f"{n_stages}",
            f"≈ {n_equiv:.2f} equivalentes"
        )

        c4.metric(
            "Pinch",
            f"x = {xp:.3f}"
        )

        # ====================================================
        # INFORMACIÓN TERMODINÁMICA
        # ====================================================

        with st.expander(
            "🌡️ Información termodinámica"
        ):

            thermo = pd.DataFrame({
                "Propiedad": [
                    f"T saturación {lk}",
                    f"T saturación {hk}",
                    "T mínima curva",
                    "T máxima curva",
                    "Presión"
                ],

                "Valor": [
                    f"{T_bubble_pure_lk:.2f} °C",
                    f"{T_bubble_pure_hk:.2f} °C",
                    f"{T_eq.min():.2f} °C",
                    f"{T_eq.max():.2f} °C",
                    f"{P_col:.2f} bar"
                ]
            })

            st.dataframe(
                thermo,
                hide_index=True,
                use_container_width=True
            )

        # ====================================================
        # GRÁFICA
        # ====================================================

        col_plot, col_data = st.columns(
            [2.1, 1]
        )

        with col_plot:

            fig = go.Figure()

            # Equilibrio
            fig.add_trace(
                go.Scatter(
                    x=x_eq,
                    y=y_eq,
                    mode="lines",
                    name="Equilibrio",
                    line=dict(
                        width=3
                    )
                )
            )

            # Diagonal
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Diagonal",
                    line=dict(
                        dash="dash"
                    )
                )
            )

            # Línea q
            if abs(q_val - 1.0) < 1e-10:

                fig.add_trace(
                    go.Scatter(
                        x=[zf, zf],
                        y=[
                            zf,
                            y_int
                        ],
                        mode="lines",
                        name="Línea q"
                    )
                )

            else:

                xq = np.linspace(
                    x_int,
                    zf,
                    100
                )

                yq = q_line(
                    xq,
                    zf,
                    q_val
                )

                fig.add_trace(
                    go.Scatter(
                        x=xq,
                        y=yq,
                        mode="lines",
                        name="Línea q"
                    )
                )

            # Rectificación
            fig.add_trace(
                go.Scatter(
                    x=[
                        x_int,
                        xd
                    ],
                    y=[
                        y_int,
                        xd
                    ],
                    mode="lines",
                    name="Rectificación",
                    line=dict(
                        width=2
                    )
                )
            )

            # Agotamiento
            fig.add_trace(
                go.Scatter(
                    x=[
                        xb,
                        x_int
                    ],
                    y=[
                        xb,
                        y_int
                    ],
                    mode="lines",
                    name="Agotamiento",
                    line=dict(
                        width=2
                    )
                )
            )

            # Pinch
            fig.add_trace(
                go.Scatter(
                    x=[xp],
                    y=[yp],
                    mode="markers",
                    name="Pinch",
                    marker=dict(
                        size=10
                    )
                )
            )

            # Escalones
            fig.add_trace(
                go.Scatter(
                    x=stage_x,
                    y=stage_y,
                    mode="lines",
                    line_shape="hv",
                    name="Etapas",
                    line=dict(
                        width=1.5
                    )
                )
            )

            fig.update_layout(
                height=650,

                xaxis_title=(
                    f"x — fracción molar líquida de {lk}"
                ),

                yaxis_title=(
                    f"y — fracción molar vapor de {lk}"
                ),

                xaxis=dict(
                    range=[0, 1]
                ),

                yaxis=dict(
                    range=[0, 1]
                ),

                hovermode="closest",

                margin=dict(
                    l=40,
                    r=30,
                    t=35,
                    b=40
                ),

                legend=dict(
                    orientation="h"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ====================================================
        # TABLA
        # ====================================================

        with col_data:

            st.markdown(
                "#### 📋 Etapas calculadas"
            )

            st.dataframe(
                df_stages,
                height=500,
                use_container_width=True,
                hide_index=True
            )

            csv_data = (
                df_stages
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "📥 Descargar CSV",
                data=csv_data,
                file_name=(
                    f"etapas_{lk}_{hk}.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"No se pudo completar el cálculo: {error}"
        )


# ============================================================
# 13. FLASH RACHFORD-RICE
# ============================================================

with tab_flash:

    st.subheader(
        f"Cálculo Flash: {lk} – {hk}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        z_fl = st.number_input(
            f"z ({lk})",
            min_value=0.001,
            max_value=0.999,
            value=0.30,
            step=0.01
        )

    with col2:

        P_fl = st.number_input(
            "Presión Flash (bar)",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1
        )

    with col3:

        psi_fl = st.slider(
            "Fracción vaporizada V/F",
            0.01,
            0.99,
            0.40,
            0.01
        )

    if st.button(
        "▶ Ejecutar simulación Flash",
        type="primary"
    ):

        try:

            (
                T_flash,
                x_flash,
                y_flash
            ) = flash_z_P_psi(
                z_fl,
                P_fl,
                psi_fl,
                ant_lk,
                ant_hk
            )

            st.success(
                f"Temperatura de equilibrio: "
                f"{T_flash:.2f} °C"
            )

            results = pd.DataFrame({

                "Componente": [
                    lk,
                    hk
                ],

                "z alimentación": [
                    z_fl,
                    1.0 - z_fl
                ],

                "x líquido": x_flash,

                "y vapor": y_flash
            })

            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True
            )

            # ------------------------------------------------
            # Comprobaciones
            # ------------------------------------------------

            liquid_sum = np.sum(x_flash)
            vapor_sum = np.sum(y_flash)

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Σx",
                f"{liquid_sum:.6f}"
            )

            c2.metric(
                "Σy",
                f"{vapor_sum:.6f}"
            )

            c3.metric(
                "V/F",
                f"{psi_fl:.2f}"
            )

            # ------------------------------------------------
            # K-values
            # ------------------------------------------------

            K_flash = K_values(
                T_flash,
                P_fl,
                ant_lk,
                ant_hk
            )

            with st.expander(
                "🔬 Detalles del equilibrio"
            ):

                equilibrium = pd.DataFrame({

                    "Componente": [
                        lk,
                        hk
                    ],

                    "K": K_flash,

                    "x": x_flash,

                    "y": y_flash
                })

                st.dataframe(
                    equilibrium,
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"No se pudo completar el Flash: {error}"
            )