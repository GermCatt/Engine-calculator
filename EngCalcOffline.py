import matplotlib.pyplot as plt
from math import pi
import attitude
import numpy as np


def test_mass(mf, af, gf, ff, kf, tf):
    m_if = mf
    for i_ in range(len(tf) - 1):
        dtf = tf[i_ + 1] - tf[i_]
        m_if = m_if - (m_if * (af[i_] + gf[i_]) + ff[i_]) / kf * (dtf / 1000)
    return m_if


while True:  # Для работы программы необходимо задать данные о времени работы двигателя. Эти данные можно получить в режиме 2. Основной режим работы - 1
    tip = input('Выполнить расчёт ИЛИ Получить диапазон? (1/2):  ')
    if tip == '1' or tip == '2':
        break
    else:
        print('Неверный ввод')
while True:  # Определение пути к файлу с данными. Формат данных: время, ускорения (xyz), угловые скорости (xyz), высота, напряжение на аккумуляторе, температура, давление.
    file = input('Укажите полный путь к файлу:  ').replace('\\', '/').replace('''"''', '').replace("'", '')
    try:
        open(file, 'r')
        break
    except FileNotFoundError:
        print('Такого файла нет или путь указан неверно')
    except PermissionError:
        print('В доступе отказано')
print('Чтение файла...')
t, ax, ay, az, wx, wy, wz, h, temp, pres = [], [], [], [], [], [], [], [], [], []
flag = True
with open(file, 'r') as f:  # запись нужных данных из файла в списки
    for line in f:
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

if tip == '2':  # Второй режим. Построение графика, по которому необходимо определить диапазон и какая ось является продольной.
    plt.plot(t, h, label='h(t)')
    plt.plot(t, ax, label='ax(t)')
    plt.plot(t, ay, label='ay(t)')
    plt.plot(t, az, label='az(t)')
    plt.legend(loc='best')
    plt.show()  # Если график не загружается - уменьшите его длину


elif tip == '1':  # Первый режим работы - определение тяги.
    print(
        'Для получения данных о диапазоне времени работы двигателя и направлении продольной оси, необходимо воспользоваться функцией получения диапазона')
    while True:  # Сообщение пользователем данных об временном интервале работы двигателя
        try:
            t_start = int(input('Время НАЧАЛА работы двигателя (мс):  '))
            t_stop = int(input('Время ОКОНЧАНИЯ работы двигателя (мс):  '))
            if t_start >= t_stop:
                print('Стартовое время больше конечного!')
            else:
                break
        except ValueError:
            print('Неверный ввод')
    while True:  # Сообщение пользователем данных о направлении продольной оси (x/y/z)
        asix = input('Какая ось направленна вдоль оси ракеты? (x/y/z):  ')
        if asix == 'x' or asix == 'y' or asix == 'z':
            break
        else:
            print('Неверный ввод')
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
    acs, omgs = [], []
    for i in range(len(ax)):
        acs.append(np.array([ax[i], ay[i], az[i]]))
        omgs.append(np.array([wx[i], wy[i], wz[i]]))
    att = attitude.Attitude(0.0075)
    att.calculate(acs, omgs)
    axn, ayn, azn = [], [], []
    for i in att.get_accs():
        axn.append(i[0])
        ayn.append(i[1])
        azn.append(i[2])
    ind = 'xyz'.index(asix)
    g = [-1 * i[ind] for i in att.get_gs()]
    a = [axn, ayn, azn][ind]
    input('Уточните время окончания работы двигателя (нажмите Enter)')
    plt.plot(t, a, label=f'a{'xyz'[ind]}(t)')
    plt.legend(loc='best')
    plt.show()
    t_stop = int(input('НОВОЕ время окончания работы двигателя (мс):  '))
    for i in range(len(t)):
        if t[i] > t_stop:
            ind_stop = i - 1
            break
    t = t[:ind_stop]
    a = a[:ind_stop]
    g = g[:ind_stop]
    v = [0]
    for i in range(len(t) - 1):  # Определяем скорость от времени
        dt = (t[i + 1] - t[i]) / 1000
        v.append(v[i] + a[i] * dt)
    while True:  # Сообщение пользователем данных о массе ракеты и топлива
        try:
            m_st = float(input('Введите стартовую массу ракеты вместе с топливом (кг):  '))
            m_t = float(input('Введите массу топлива в двигателе (кг):  '))
            if m_t >= m_st:
                print('Масса топлива больше полной массы ракеты!')
            else:
                break
        except ValueError:
            print('Неверный ввод')
    while True:  # Ввод пользователем коэффициента лобового сопротивления и калибра
        try:
            cx = float(input('Введите коэффициент базового лобового сопротивления (берётся из OR):  '))
            d = float(input('Введите калибр ракеты (мм):  '))
            break
        except ValueError:
            print('некорректный ввод')
    S = pi * (d / 1000) ** 2 / 4
    Ro = pres[ind_start - 50] * 0.029 / (8.31 * (temp[ind_start - 50] + 273))
    f_w = [0]
    for v_i in v:  # Определение значений продольной скорости ракеты
        f_w.append(v_i ** 2 * Ro * S * cx / 2)
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
    for i in range(len(t) - 1):  # Определение силы тяги и полного импульса
        dt = (t[i + 1] - t[i]) / 1000
        F.append(m_i * (a[i] + g[i]) + f_w[i])
        p += (m_i * (a[i] + g[i]) + f_w[i]) * dt
        m_i = m_i - (m_i * (a[i] + g[i]) + f_w[i]) / k * dt
    print(f'\nПолный импульс двигателя:  {p} Н*с')
    input('Для продолжения нажмите Enter')
    plt.plot(list(map(lambda rr: (rr - t[0]) / 1000, t)), F)
    plt.title('Профиль тяги')
    plt.xlabel('Время, с')
    plt.ylabel('Тяга, Н')
    plt.show()
