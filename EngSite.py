import streamlit as st
import plotly.graph_objects as go
from math import pi
import attitude
import numpy as np
import io


def test_mass(mf, af, gf, ff, kf, tf):  # Функция, выполняющая все итерации метода Эйлера для расчёта конечной массы в зависимости от k
    m_if = mf
    for i_ in range(len(tf) - 1):
        dtf = tf[i_ + 1] - tf[i_]
        m_if = m_if - (m_if * (af[i_] + gf[i_]) + ff[i_]) / kf * (dtf / 1000)   # Один шаг в методе эйлера. Переход от m_i к m_i+1
    return m_if


@st.cache_data(show_spinner="Чтение и обработка файла...")
def parse_uploaded_file(file_bytes: bytes):
    t, ax, ay, az = [], [], [], []
    wx, wy, wz = [], [], []
    h, temp, pres = [], [], []
    with io.TextIOWrapper(io.BytesIO(file_bytes), encoding="utf-8") as f:
        next(f)
        for line in f:
            values = line.strip().split(',')
            t.append(float(values[0]))
            ax.append(float(values[1]))
            ay.append(float(values[2]))
            az.append(float(values[3]))
            wx.append(float(values[4]))
            wy.append(float(values[5]))
            wz.append(float(values[6]))
            h.append(float(values[7]))
            temp.append(float(values[9]))
            pres.append(float(values[10]))

    return t, ax, ay, az, wx, wy, wz, h, temp, pres


st.title("Калькулятор характеристик двигателя")
st.write("Данный калькулятор позволяет рассчитать полный импульс двигателя, а также тяговый профиль двигателя, базируясь на данных с бортового самописца, а также на геометрических характеристиках ракеты")

uploaded_file = st.file_uploader("Чтение файлов телеметрии. Формат: время (с), ускорения по трём осям (м/с^2), угловая скорость по трём осям (рад/c), высота (м), температура (°C), давление (Па)", type="txt")
t = []
if uploaded_file:
    try:
        file_bytes = uploaded_file.getvalue()
        t, ax, ay, az, wx, wy, wz, h, temp, pres = parse_uploaded_file(file_bytes)
    except Exception as e:
        print(f'Ошибка чтения файла: {e}')
if t:
    try:
        del uploaded_file
        uploaded_file = 0
        t0 = 0
        for i in range(len(h)):     # Сокращение диапазона данных для оптимизации построения графика
            if h[i] >= 100:
                t0 = t[i]
                break
        tmin = t0 - 30000
        tmax = t0 + 130000
        imin = 0
        imax = len(t)
        if len(t) >= 7500:
            for i in range(len(t) - 1):  # Определяем начальный и конечный индекс для сокращения списков
                if t[i] >= tmin and not imin:
                    imin = i
                elif t[i] > tmax:
                    imax = i - 1
                    break
            # Сокращение списков
            t = t[imin:imax]
            ax, ay, az = ax[imin:imax], ay[imin:imax], az[imin:imax]
            wx, wy, wz = wx[imin:imax], wy[imin:imax], wz[imin:imax]
            h, temp, pres = h[imin:imax], temp[imin:imax], pres[imin:imax]
        st.write("Обработка файла завершена")
        # Задаём все графики и их параметры
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=h,
            name="h(t)",
            line=dict(color='blue', width=2),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=t, y=ax,
            name="ax(t)",
            line=dict(color='red', width=2),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=t, y=ay,
            name="ay(t)",
            line=dict(color='green', width=2),
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=t, y=az,
            name="az(t)",
            line=dict(color='orange', width=2),
            mode='lines'
        ))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
    t_start = st.number_input(  # Ввод пользователем стартовой массы ракеты
        "Время начала работы двигателя (мс)",
        min_value=0.0,
        key='num1',
        step=1.0
    )
    t_stop = 0
    if t_start:
        t_stop = st.number_input(  # Ввод пользователем стартовой массы ракеты
            "Время окончания работы двигателя (мс)",
            min_value=float(t_start),
            key='num2',
            step=1.0
        )
    asix = st.selectbox(  # Задаём направление продольной оси с возможностью его изменения пользователем
        "Какая ось является продольной осью ракеты?",
        ["x", "y", "z"],
        key='num3'
    )
    fig3 = go.Figure()
    if all([t_start, t_stop > t_start, asix]):
        fig3.add_trace(go.Scatter(
            x=t, y=[ax, ay, az]['xyz'.index(asix)],
            name=f"a{'xyz'['xyz'.index(asix)]}(t)",
            line=dict(color='blue', width=2),
            mode='lines'
        ))
        fig3.add_trace(go.Scatter(
            x=[t_start] * 2, y=[min([ax, ay, az]['xyz'.index(asix)]), max([ax, ay, az]['xyz'.index(asix)])],
            line=dict(dash='dash', color='red', width=2),
            name='Диапазон',
            mode='lines'
        ))
        fig3.add_trace(go.Scatter(
            x=[t_stop] * 2, y=[min([ax, ay, az]['xyz'.index(asix)]), max([ax, ay, az]['xyz'.index(asix)])],
            line=dict(dash='dash', color='red', width=2),
            name='Диапазон',
            mode='lines'
        ))
        fig3.update_layout(
            title="Кажущиеся ускорения вдоль выбранной оси и введённый диапазон времени работы двигтаеля",
            xaxis_title="Время, мс",
            yaxis_title="Ускорение, м/c^2"
        )
        st.plotly_chart(fig3, use_container_width=True)
    m_st = st.number_input(     # Ввод пользователем стартовой массы ракеты
        "Стартовая масса ракеты вместе с топливом (кг)",
        min_value=0.0,
        key='num4',
        step=0.001,
        format="%.3f"
    )
    m_t = 0
    if m_st:
        m_t = st.number_input(      # Ввод пользователем массы топлива
            "Масса топлива (кг)",
            max_value=m_st,
            key='num5',
            step=0.001,
            format="%.3f"
        )
    cx = st.number_input(       # Ввод пользователем коэффициента базового лобового сопротивления
        "Коэффициент базового лобового сопротивление (берётся из Open Rocket)",
        min_value=0.0,
        key='num6',
        step=0.001,
        format="%.3f"
    )
    d = st.number_input(        # Ввод пользователем диаметра сечения миделя
        "Диаметр сечения миделя (мм)",
        min_value=0.0,
        key='num7'
    )

    if all([m_st, m_t, cx, d]):
        if st.button("Выполнить расчёты", key="calculate"):
            ind_start = 0
            ind_stop = 0
            for i in range(len(t)):  # Определяем начальный и конечный индекс для оптимизации расчёта
                if t[i] >= t_start and not ind_start:
                    ind_start = i
                elif t[i] > t_stop:
                    ind_stop = i - 1
                    break
            # Пересоздаём списки для удобства
            t = t[ind_start:ind_stop + 1]
            ax, ay, az = ax[ind_start:ind_stop + 1], ay[ind_start:ind_stop + 1], az[ind_start:ind_stop + 1]
            wx, wy, wz = wx[ind_start:ind_stop + 1], wy[ind_start:ind_stop + 1], wz[ind_start:ind_stop + 1]
            acs, omgs = [], []
            for i in range(len(ax)):  # создание списков acs и omgs с ускорениями и угловыми скоростями в нужном формате
                acs.append(np.array([ax[i], ay[i], az[i]]))
                omgs.append(np.array([wx[i], wy[i], wz[i]]))
            dtimes = []
            for i in range(len(t) - 1):
                dtimes.append((t[i + 1] - t[i]) / 1000)
            dtimes.append(dtimes[0])
            att = attitude.Attitude(0.007)
            att.calculate(dtimes, acs, omgs)
            axn, ayn, azn = [], [], []
            for i in att.get_accs():  # Получение данных, посчитанных attitude и создание списков ускорений по осям ракеты
                axn.append(i[0])
                ayn.append(i[1])
                azn.append(i[2])
            ind = 'xyz'.index(asix)
            g = [-1 * i[ind] for i in
                 att.get_gs()]  # Создание списка ускорений свободного падения в проекции на ось ракеты, посчитанных attitude
            a = [axn, ayn, azn][ind]
            ind_stop2 = ind_stop
            ind_start2 = ind_start
            for i in range(len(a) // 2):
                if a[i + 1] - a[i] > 1.5:
                    ind_start2 = i
                    break
            for i in range(len(a) // 2, len(a)):  # Получение реального момента прекращения работы двигателя
                if a[i] < -1 * g[i]:
                    ind_stop2 = i
                    break
            # Пересоздаём списки для удобства
            t = t[ind_start2:ind_stop2 + 1]
            a = a[ind_start2:ind_stop2 + 1]
            g = g[ind_start2:ind_stop2 + 1]
            S = pi * (d / 1000) ** 2 / 4
            Ro = pres[ind_start - 50] * 0.029 / (8.31 * (temp[
                                                             ind_start - 50] + 273))  # Плотность воздуха, получается через уравнение Менделева-Клапейрона
            v = [0]
            f_w = [0]
            for i in range(len(t) - 1):
                dt = (t[i + 1] - t[i]) / 1000
                v.append(
                    v[i] + a[i] * dt)  # Скорость в каждый момент времени через интегрирование ускорений методом Эйлера
                f_w.append(v[
                               i] ** 2 * Ro * S * cx / 2)  # Сила сопротивления воздуха в каждый момент времени через скорость и вводимые параметры
            k_high = 10000
            k_low = 1
            k_mid = (k_low + k_high) / 2
            m_tol = m_st * 10 ** -7  # Окрестность для метода бисекции
            it = 0
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
            for i in range(len(t) - 1):  # Определение силы тяги и полного имп ульса
                dt = (t[i + 1] - t[i]) / 1000
                F.append(m_i * (a[i] + g[i]) + f_w[i])
                p += (m_i * (a[i] + g[i]) + f_w[i]) * dt
                m_i = m_i - (m_i * (a[i] + g[i]) + f_w[i]) / k * dt

            st.success("Расчёт завершён")

            # Настройка графика тягового профиля двигателя
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(map(lambda aa: (aa - t[0]) / 1000, t)), y=F,
                name="F(t)",
                line=dict(color='red', width=2),
                mode='lines'
            ))
            fig2.update_layout(
                title=f"Тяговый профиль двигателя, полный импульс {round(p, 3)} Н*с",
                xaxis_title="Время, с",
                yaxis_title="Тяга, Н"
            )
            st.plotly_chart(fig2, use_container_width=True)