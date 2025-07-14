import streamlit as st
import plotly.graph_objects as go
from math import pi
from io import BytesIO
import kaleido


def test_mass(mf, af, gf, ff, kf, tf):
    m_if = mf
    for i_ in range(len(tf) - 1):
        dtf = tf[i_ + 1] - tf[i_]
        m_if = m_if - (m_if * (af[i_] + gf[i_]) + ff[i_]) / kf * (dtf / 1000)
    return m_if


st.title("Калькулятор характеристик двигателя")
st.write("Данный калькулятор позволяет рассчитать полный импульс двигателя, а также тяговый профиль двигателя, базируясь на данных с бортового самописца, а также на геометрических характеристиках ракеты")

uploaded_file = st.file_uploader("Чтение файлов телеметрии. Формат: время (с), ускорения по трём осям (м/с^2), угловая скорость по трём осям (рад/c), высота (м), температура (°C), давление (Па)", type="txt")

if uploaded_file is not None:
    t, ax, ay, az, wx, wy, wz, h, temp, pres = [], [], [], [], [], [], [], [], [], []
    file_content = uploaded_file.read().decode("utf-8")
    lines = file_content.splitlines()
    flag = True
    st.write("Файл обрабатывается")
    for line in lines:
        if flag:
            flag = False
            continue
        _ = line[:-2].split(sep=',')
        t.append(float(_[0]))
        ax.append(float(_[1]))
        ay.append(float(_[2]))
        az.append(float(_[3]))
        wx.append((float(_[4])))
        wy.append(float(_[5]))
        wz.append(float(_[6]))
        h.append(float(_[7]))
        temp.append(float(_[9]))
        pres.append(float(_[10]))
    st.write("Обработка файла завершена")
    if st.session_state.get("show_plot", False):
        x = t
        y1 = h
        y2 = ax
        y3 = ay
        y4 = az

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y1,
            name="h(t)",
            line=dict(color='blue', width=2),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=x, y=y2,
            name="ax(t)",
            line=dict(color='red', width=2, dash='dash'),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=x, y=y3,
            name="ay(t)",
            line=dict(color='green', width=3),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y4,
            name="az(t)",
            line=dict(color='orange', width=3),
            mode='lines'
        ))
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Свернуть график (повысит производительность)"):
            st.session_state.show_plot = False
            st.rerun()
    else:
        if st.button("Развернуть график"):
            st.session_state.show_plot = True
            st.rerun()

    t_start = st.number_input(
        "Время НАЧАЛА работы двигателя (мс)",
        min_value=0.0,
        step=1.0,
        key='num1'
    )
    t_stop = 0
    if t_start:
        t_stop = st.number_input(
            "Время ОКОНЧАНИЯ работы двигателя (мс)",
            min_value=t_start,
            step=1.0,
            key='num2'
        )

    asix = st.selectbox(
        "Какая ось является продольной осью ракеты?",
        ["x", "y", "z"],
        index=0,
        key='num3'
    )
    m_st = st.number_input(
        "Стартовая масса ракеты вместе с топливом (кг)",
        min_value=0.0,
        key='num4'
    )
    m_t = 0
    if m_st:
        m_t = st.number_input(
            "Масса топлива (кг)",
            max_value=m_st,
            key='num5'
        )
    cx = st.number_input(
        "Коэффициент базового лобового сопротивление (берётся из Open Rocket)",
        min_value=0.0,
        key='num6'
    )
    d = st.number_input(
        "Диаметр сечения миделя (мм)",
        min_value=0.0,
        key='num7'
    )
    if t_start and t_stop and cx and d and m_st and m_t:
        ind_start = 0
        ind_stop = 0
        for i in range(len(t)):  # Определяем начальный и конечный индекс для оптимизации расчёта
            if t[i] >= t_start and not ind_start:
                ind_start = i
            elif t[i] > t_stop:
                ind_stop = i - 1
                break
        # Пересоздаём списки для экономии памяти, отсекаем лишнее
        t = t[ind_start:ind_stop + 1]
        ax, ay, az = ax[ind_start:ind_stop + 1], ay[ind_start:ind_stop + 1], az[ind_start:ind_stop + 1]
        wx, wy, wz = wx[ind_start:ind_stop + 1], wy[ind_start:ind_stop + 1], wz[ind_start:ind_stop + 1]
        a = [ax, ay, az]['xyz'.index(asix)]
        w_kr = [wx, wy, wz]['xyz'.index(asix)]
        _ = [wx, wy, wz]
        _.pop('xyz'.index(asix))
        w_tan, w_rys = tuple(_)
        S = pi * (d / 1000) ** 2 / 4
        Ro = pres[ind_start - 50] * 0.029 / (8.31 * (temp[ind_start - 50] + 273))
        v = [0]
        f_w = [0]
        for i in range(len(t) - 1):
            dt = (t[i + 1] - t[i]) / 1000
            v.append(v[i] + a[i] * dt)
            f_w.append(v[i] ** 2 * Ro * S * cx / 2)
        g = [9.806 for _ in range(len(t))]  # ПЛЕЙСХОЛДЕР УЧЕСТЬ НАКЛОН ПОЗЖЕ
        k_high = 10000
        k_low = 1
        k_mid = (k_low + k_high) / 2
        m_tol = m_st * 10 ** -7  # Окрестность для метода бисекции
        it = 0
        if st.button("Выполнить расчёты", key="calculate"):
            while True:  # Определение коэффициента пропорциональности k методом бисекции
                it += 1
                if (test_mass(m_st, a, g, f_w, k_low, t) - m_st + m_t) * (
                        test_mass(m_st, a, g, f_w, k_high, t) - m_st + m_t) > 0:
                    print('ERROR', it)
                    break
                k_mid = (k_low + k_high) / 2
                if abs(test_mass(m_st, a, g, f_w, k_mid, t) - m_st + m_t) < m_tol:
                    break
                elif (test_mass(m_st, a, g, f_w, k_mid, t) - m_st + m_t) * (
                        test_mass(m_st, a, g, f_w, k_low, t) - m_st + m_t) < 0:
                    k_high = k_mid
                else:
                    k_low = k_mid
            k = k_mid
            m_i = m_st
            F = [0]
            p = 0
            for i in range(len(t) - 1):  # Определение силы тяги и полного импульса
                dt = (t[i + 1] - t[i]) / 1000
                F.append(m_i * (a[i] + g[i]) + f_w[i])
                p += (m_i * (a[i] + g[i]) + f_w[i]) * dt
                m_i = m_i - (m_i * (a[i] + g[i]) + f_w[i]) / k * dt

            st.success("Расчёт завершён")
            st.write(f"Полный импульс двигателя {p} Н*с")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(map(lambda aa: (aa - t[0]) / 1000, t)), y=F,
                name="F(t)",
                line=dict(color='red', width=2),
                mode='lines'
            ))
            fig2.update_layout(
                title="Тяговый профиль двигателя",
                xaxis_title="Время, с",
                yaxis_title="Тяга, Н"
            )
            st.plotly_chart(fig2, use_container_width=True)
            buf = BytesIO()
            fig2.write_image(buf, format="png", engine="kaleido")
            buf.seek(0)
            st.download_button(
                "Скачать график",
                data=buf,
                file_name="plot.png",
                mime="image/png",
                key="download_plot"
            )