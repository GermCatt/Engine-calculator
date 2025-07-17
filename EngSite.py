import streamlit as st
import plotly.graph_objects as go
from math import pi
import attitude
import numpy as np
from streamlit_plotly_events import plotly_events

@st.cache_resource
def calc_attitude(dtimes, acs, omgs):
    att = attitude.Attitude(0.007)
    att.calculate(dtimes, acs, omgs)
    return att

def test_mass(mf, af, gf, ff, kf, tf):
    m_if = mf
    for i_ in range(len(tf) - 1):
        dtf = tf[i_ + 1] - tf[i_]
        m_if = m_if - (m_if * (af[i_] + gf[i_]) + ff[i_]) / kf * (dtf / 1000)
    return m_if

st.title("Калькулятор характеристик двигателя")
st.write("Калькулятор для расчёта импульса и тягового профиля по данным самописца и геометрии ракеты")

uploaded_file = st.file_uploader("Загрузите файл телеметрии", type="txt")
if uploaded_file:
    try:
        t, ax, ay, az, wx, wy, wz, h, temp, pres = [], [], [], [], [], [], [], [], [], []
        file_content = uploaded_file.read().decode("utf-8")
        lines = file_content.splitlines()
        for line in lines[1:]:
            _ = line[:-2].split(sep=',')
            t.append(float(_[0]))
            ax.append(float(_[1]))
            ay.append(float(_[2]))
            az.append(float(_[3]))
            wx.append(float(_[4]))
            wy.append(float(_[5]))
            wz.append(float(_[6]))
            h.append(float(_[7]))
            temp.append(float(_[9]))
            pres.append(float(_[10]))

        t0 = next((t[i] for i in range(len(h)) if h[i] >= 100), 0)
        tmin, tmax = t0 - 30000, t0 + 130000
        imin = next(i for i in range(len(t)) if t[i] >= tmin)
        imax = next((i for i in range(imin, len(t)) if t[i] > tmax), len(t) - 1)

        t = t[imin:imax]
        ax, ay, az = ax[imin:imax], ay[imin:imax], az[imin:imax]
        wx, wy, wz = wx[imin:imax], wy[imin:imax], wz[imin:imax]
        h, temp, pres = h[imin:imax], temp[imin:imax], pres[imin:imax]

        if 'clicks' not in st.session_state:
            st.session_state.clicks = []
        if 'ignore_next_click' not in st.session_state:
            st.session_state.ignore_next_click = False

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=h, name="h(t)", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=t, y=ax, name="ax(t)", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=t, y=ay, name="ay(t)", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=t, y=az, name="az(t)", line=dict(color='orange')))
        fig.update_layout(title='Полётные характеристики', clickmode='event+select')

        clicked = plotly_events(fig, click_event=True, hover_event=False)
        if clicked and len(st.session_state.clicks) < 2:
            if st.session_state.ignore_next_click:
                st.session_state.ignore_next_click = False
            else:
                st.session_state.clicks.append(clicked[0])

        if st.button("Сбросить выбор"):
            st.session_state.clicks = []
            st.session_state.ignore_next_click = True
            st.rerun()

        if len(st.session_state.clicks) < 2:
            st.warning("Выберите точку начала и конца работы двигателя")
            st.stop()

        t_start, t_stop = sorted([st.session_state.clicks[0]['x'], st.session_state.clicks[1]['x']])
        st.write(f"Начало работы двигателя: {t_start}")
        st.write(f"Окончание работы двигателя: {t_stop}")

        with st.form("input_form"):
            asix_base = st.session_state.clicks[0]['curveNumber']
            asix = st.selectbox("Продольная ось ракеты", ["x", "y", "z"], index=asix_base - 1)
            m_st = st.number_input("Стартовая масса ракеты (кг)", min_value=0.0, step=0.001, format="%.3f")
            m_t = st.number_input("Масса топлива (кг)", min_value=0.0, max_value=m_st, step=0.001, format="%.3f")
            cx = st.number_input("Коэффициент лобового сопротивления", min_value=0.0, step=0.001, format="%.3f")
            d = st.number_input("Диаметр миделя (мм)", min_value=0.0)
            submitted = st.form_submit_button("Рассчитать")

        if submitted:
            with st.spinner("Обработка данных..."):
                ind_start = next(i for i in range(len(t)) if t[i] >= t_start)
                ind_stop = next(i for i in range(ind_start, len(t)) if t[i] > t_stop) - 1
                t = t[ind_start:ind_stop + 1]
                ax, ay, az = ax[ind_start:ind_stop + 1], ay[ind_start:ind_stop + 1], az[ind_start:ind_stop + 1]
                wx, wy, wz = wx[ind_start:ind_stop + 1], wy[ind_start:ind_stop + 1], wz[ind_start:ind_stop + 1]
                acs = [np.array([ax[i], ay[i], az[i]]) for i in range(len(ax))]
                omgs = [np.array([wx[i], wy[i], wz[i]]) for i in range(len(wx))]
                dtimes = [(t[i + 1] - t[i]) / 1000 for i in range(len(t) - 1)] + [0.01]

                att = calc_attitude(dtimes, acs, omgs)
                axn, ayn, azn = zip(*att.get_accs())
                ind = 'xyz'.index(asix)
                a = [axn, ayn, azn][ind]
                g = [-1 * gs[ind] for gs in att.get_gs()]
                t_stop2 = next((t[i] for i in range(len(a) // 2, len(a)) if a[i] < 0), t[-1])

                if all([t_stop2, m_t, cx, d]):
                    ind_stop2 = next((i for i in range(len(t)) if t[i] > t_stop2), len(t) - 1)
                    t = t[:ind_stop2]
                    a = a[:ind_stop2]
                    g = g[:ind_stop2]
                    S = pi * (d / 1000) ** 2 / 4
                    Ro = pres[ind_start - 50] * 0.029 / (8.31 * (temp[ind_start - 50] + 273))
                    v, f_w = [0], [0]
                    for i in range(len(t) - 1):
                        dt = (t[i + 1] - t[i]) / 1000
                        v.append(v[i] + a[i] * dt)
                        f_w.append(v[i] ** 2 * Ro * S * cx / 2)

                    k_low, k_high = 1, 10000
                    m_tol = m_st * 1e-7
                    while True:
                        k_mid = (k_low + k_high) / 2
                        diff = test_mass(m_st, a, g, f_w, k_mid, t) - m_st + m_t
                        if abs(diff) < m_tol:
                            break
                        if diff * (test_mass(m_st, a, g, f_w, k_low, t) - m_st + m_t) < 0:
                            k_high = k_mid
                        else:
                            k_low = k_mid

                    k = k_mid
                    m_i = m_st
                    F, p = [0], 0
                    for i in range(len(t) - 1):
                        dt = (t[i + 1] - t[i]) / 1000
                        thrust = m_i * (a[i] + g[i]) + f_w[i]
                        F.append(thrust)
                        p += thrust * dt
                        m_i -= thrust / k * dt

                    st.success("Расчёт завершён")
                    st.write(f"Полный импульс двигателя: {p:.3f} Н*с")

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=[(tt - t[0]) / 1000 for tt in t], y=F, name="F(t)", line=dict(color='red')))
                    fig2.update_layout(title="Тяговый профиль двигателя", xaxis_title="Время, с", yaxis_title="Тяга, Н")
                    st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
