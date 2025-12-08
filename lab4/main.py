import itertools
import numpy as np
from collections import defaultdict
import time

# ------------------------------
# 1. ИСХОДНЫЕ ДАННЫЕ С КОМИССИЯМИ
# ------------------------------
S1_0, S2_0, D_0, C_0 = 100, 800, 400, 600
MIN_S1, MIN_S2, MIN_D = 30, 150, 100

# Комиссии брокеров
COM_S1 = 0.04  # 4%
COM_S2 = 0.07  # 7%
COM_D = 0.05  # 5%

# Коэффициенты доходности
k = {
    1: [[1.20, 1.10, 1.07], [1.05, 1.02, 1.03], [0.80, 0.95, 1.00]],
    2: [[1.40, 1.15, 1.01], [1.05, 1.00, 1.00], [0.60, 0.90, 1.00]],
    3: [[1.15, 1.12, 1.05], [1.05, 1.01, 1.01], [0.70, 0.94, 1.00]]
}

# Вероятности
p = {
    1: [0.60, 0.30, 0.10],
    2: [0.30, 0.20, 0.50],
    3: [0.40, 0.40, 0.20]
}

# Управления
U_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5]

# Шаги дискретизации
STEP_S1 = 25
STEP_S2 = 200
STEP_D = 100
STEP_C = 100


# ------------------------------
# 2. ФУНКЦИИ С УЧЕТОМ КОМИССИЙ
# ------------------------------
def discretize(s1, s2, d, c):
    """Дискретизация состояния"""
    s1 = max(MIN_S1, round(s1 / STEP_S1) * STEP_S1)
    s2 = max(MIN_S2, round(s2 / STEP_S2) * STEP_S2)
    d = max(MIN_D, round(d / STEP_D) * STEP_D)
    c = max(0, round(c / STEP_C) * STEP_C)
    return (s1, s2, d, c)


def apply_management_with_commission(s1, s2, d, c, u1, u2, ud):
    """Применение управления с учетом комиссий"""
    # Суммы операций
    amount_s1 = u1 * S1_0
    amount_s2 = u2 * S2_0
    amount_d = ud * D_0

    # Комиссии
    commission_cost = 0
    if amount_s1 > 0:  # покупка ЦБ1
        commission_cost += amount_s1 * COM_S1
    elif amount_s1 < 0:  # продажа ЦБ1
        commission_cost += abs(amount_s1) * COM_S1

    if amount_s2 > 0:  # покупка ЦБ2
        commission_cost += amount_s2 * COM_S2
    elif amount_s2 < 0:  # продажа ЦБ2
        commission_cost += abs(amount_s2) * COM_S2

    if amount_d > 0:  # увеличение депозита
        commission_cost += amount_d * COM_D
    elif amount_d < 0:  # снятие с депозита
        commission_cost += abs(amount_d) * COM_D

    # Общая стоимость операций + комиссии
    total_cost = amount_s1 + amount_s2 + amount_d + commission_cost

    # Проверяем, хватает ли денег
    if total_cost > c + 1e-9:
        return None, None

    # Новое состояние
    new_s1 = s1 + amount_s1
    new_s2 = s2 + amount_s2
    new_d = d + amount_d
    new_c = c - total_cost

    # Проверяем минимальные остатки
    if (new_s1 < MIN_S1 - 1e-9 or new_s2 < MIN_S2 - 1e-9 or
            new_d < MIN_D - 1e-9 or new_c < -1e-9):
        return None, None

    new_state = discretize(new_s1, new_s2, new_d, new_c)
    commission_paid = commission_cost

    return new_state, commission_paid


def generate_states_focused():
    """Генерация состояний, сфокусированных на достижимых"""
    states = []

    # Фокус на начальном состоянии и его окрестностях
    center_s1, center_s2, center_d, center_c = S1_0, S2_0, D_0, C_0

    # Диапазоны вокруг начальных значений
    s1_range = np.arange(max(MIN_S1, center_s1 - 3 * STEP_S1),
                         center_s1 + 4 * STEP_S1, STEP_S1)
    s2_range = np.arange(max(MIN_S2, center_s2 - 3 * STEP_S2),
                         center_s2 + 4 * STEP_S2, STEP_S2)
    d_range = np.arange(max(MIN_D, center_d - 3 * STEP_D),
                        center_d + 4 * STEP_D, STEP_D)
    c_range = np.arange(max(0, center_c - 5 * STEP_C),
                        center_c + 10 * STEP_C, STEP_C)

    for s1 in s1_range:
        for s2 in s2_range:
            for d in d_range:
                for c in c_range:
                    states.append(discretize(s1, s2, d, c))

    # Уникальные состояния
    states = list(set(states))
    print(f"Сгенерировано {len(states)} сфокусированных состояний")
    return states


# ------------------------------
# 3. ОБРАТНАЯ ПРОГОНКА С КОМИССИЯМИ
# ------------------------------
def backward_induction_with_commissions():
    """Обратная прогонка с учетом комиссий"""
    print("Генерация состояний...")
    all_states = generate_states_focused()

    # Добавляем начальное состояние
    init_state = discretize(S1_0, S2_0, D_0, C_0)
    if init_state not in all_states:
        all_states.append(init_state)

    # F_n[state] = ожидаемый капитал
    F = {3: {}, 2: {}, 1: {}}
    U = {3: {}, 2: {}, 1: {}}

    # Шаг 1: F_4(S) = S1 + S2 + D + C
    F_next = {}
    for state in all_states:
        s1, s2, d, c = state
        if s1 >= MIN_S1 and s2 >= MIN_S2 and d >= MIN_D and c >= 0:
            F_next[state] = s1 + s2 + d + c

    print(f"Конечных состояний: {len(F_next)}")

    # Шаг 2: Обратная прогонка
    for n in [3, 2, 1]:
        print(f"\n--- Этап {n} ---")
        start_time = time.time()

        F_n = {}
        U_n = {}

        for state in all_states:
            s1, s2, d, c = state

            if s1 < MIN_S1 or s2 < MIN_S2 or d < MIN_D or c < 0:
                continue

            best_value = -1e9
            best_u = (0, 0, 0)

            for u1 in U_VALUES:
                for u2 in U_VALUES:
                    for ud in U_VALUES:
                        # Применяем управление с комиссиями
                        result = apply_management_with_commission(s1, s2, d, c, u1, u2, ud)
                        if result[0] is None:
                            continue

                        new_state, commission_paid = result
                        new_s1, new_s2, new_d, new_c = new_state

                        # Вычисляем ожидаемый доход
                        exp_val = 0
                        for j in range(3):
                            # После ситуации
                            s1_j = new_s1 * k[n][j][0]
                            s2_j = new_s2 * k[n][j][1]
                            d_j = new_d * k[n][j][2]
                            c_j = new_c - commission_paid  # комиссии уже учтены

                            state_j = discretize(s1_j, s2_j, d_j, c_j)

                            # Значение из следующего этапа
                            value = F_next.get(state_j, 0)

                            # Для последнего этапа
                            if n == 3 and value == 0:
                                value = s1_j + s2_j + d_j + c_j

                            exp_val += p[n][j] * value

                        if exp_val > best_value:
                            best_value = exp_val
                            best_u = (u1, u2, ud)

            if best_value > -1e8:
                F_n[state] = best_value
                U_n[state] = best_u

        F[n] = F_n
        U[n] = U_n
        F_next = F_n

        elapsed = time.time() - start_time
        print(f"Обработано {len(F_n)} состояний за {elapsed:.2f} сек")

    return F, U, all_states


# ------------------------------
# 4. АНАЛИЗ И ВЫВОД РЕЗУЛЬТАТОВ
# ------------------------------
def analyze_results(F, U):
    """Анализ и вывод результатов"""
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    init_state = discretize(S1_0, S2_0, D_0, C_0)
    initial_capital = S1_0 + S2_0 + D_0 + C_0

    # 1. Оптимальный ожидаемый доход
    if init_state in F[1]:
        optimal_capital = F[1][init_state]
        optimal_income = optimal_capital - initial_capital
        optimal_return = 100 * optimal_income / initial_capital
    else:
        # Ищем ближайшее
        closest = min(F[1].keys(),
                      key=lambda s: sum(abs(a - b) for a, b in zip(s, init_state)))
        optimal_capital = F[1][closest]
        optimal_income = optimal_capital - initial_capital
        optimal_return = 100 * optimal_income / initial_capital
        print(f"Используем ближайшее состояние: {closest}")

    print(f"\n1. ОПТИМАЛЬНАЯ СТРАТЕГИЯ (С УЧЕТОМ КОМИССИЙ):")
    print(f"   Начальный капитал: {initial_capital:.1f} д.е.")
    print(f"   Ожидаемый конечный капитал: {optimal_capital:.1f} д.е.")
    print(f"   Ожидаемый доход: {optimal_income:+.1f} д.е.")
    print(f"   Ожидаемая доходность: {optimal_return:+.2f}%")

    # 2. Пассивная стратегия
    print(f"\n2. ПАССИВНАЯ СТРАТЕГИЯ (без управления):")
    s1, s2, d = S1_0, S2_0, D_0
    for n in [1, 2, 3]:
        # Матожидание доходности
        exp_s1 = sum(p[n][j] * k[n][j][0] for j in range(3))
        exp_s2 = sum(p[n][j] * k[n][j][1] for j in range(3))
        exp_d = sum(p[n][j] * k[n][j][2] for j in range(3))

        s1 *= exp_s1
        s2 *= exp_s2
        d *= exp_d

    passive_capital = s1 + s2 + d + C_0
    passive_income = passive_capital - initial_capital
    passive_return = 100 * passive_income / initial_capital
    print(f"   Ожидаемый конечный капитал: {passive_capital:.1f} д.е.")
    print(f"   Ожидаемый доход: {passive_income:+.1f} д.е.")
    print(f"   Ожидаемая доходность: {passive_return:+.2f}%")

    # 3. Сравнение
    print(f"\n3. СРАВНЕНИЕ СТРАТЕГИЙ:")
    if optimal_income > passive_income:
        advantage = optimal_income - passive_income
        print(f"   ✓ Активное управление лучше на {advantage:.1f} д.е.")
        print(f"   ✓ Выигрыш: {100 * advantage / passive_capital:.2f}%")
    else:
        disadvantage = passive_income - optimal_income
        print(f"   ✗ Пассивная стратегия лучше на {disadvantage:.1f} д.е.")
        print(f"   ✗ Проигрыш: {100 * disadvantage / passive_capital:.2f}%")

    # 4. Ожидаемые доходности активов
    print(f"\n4. ОЖИДАЕМЫЕ ДОХОДНОСТИ АКТИВОВ ПО ЭТАПАМ:")
    for n in [1, 2, 3]:
        exp_s1 = sum(p[n][j] * (k[n][j][0] - 1) for j in range(3))
        exp_s2 = sum(p[n][j] * (k[n][j][1] - 1) for j in range(3))
        exp_d = sum(p[n][j] * (k[n][j][2] - 1) for j in range(3))
        print(f"   Этап {n}: ЦБ1={100 * exp_s1:+.1f}%, ЦБ2={100 * exp_s2:+.1f}%, Деп={100 * exp_d:+.1f}%")

    # 5. Рекомендации
    print(f"\n5. РЕКОМЕНДАЦИИ:")

    # Находим оптимальный путь
    path = []
    s1, s2, d, c = S1_0, S2_0, D_0, C_0

    for n in [1, 2, 3]:
        state = discretize(s1, s2, d, c)

        # Ищем ближайшее состояние
        closest = min(U[n].keys(),
                      key=lambda s: sum(abs(a - b) for a, b in zip(s, state)))

        u1, u2, ud = U[n][closest]
        path.append((n, u1, u2, ud))

        # Обновляем состояние для следующего этапа
        s1 += u1 * S1_0
        s2 += u2 * S2_0
        d += ud * D_0
        c -= (u1 * S1_0 + u2 * S2_0 + ud * D_0)

    # Выводим рекомендации
    for n, u1, u2, ud in path:
        print(f"   Этап {n}:", end=" ")
        actions = []
        if abs(u1) > 0.01:
            actions.append(f"ЦБ1: {'купить' if u1 > 0 else 'продать'} {abs(u1):.2f} доли")
        if abs(u2) > 0.01:
            actions.append(f"ЦБ2: {'купить' if u2 > 0 else 'продать'} {abs(u2):.2f} доли")
        if abs(ud) > 0.01:
            actions.append(f"Деп: {'пополнить' if ud > 0 else 'снять'} {abs(ud):.2f} доли")

        if actions:
            print(", ".join(actions))
        else:
            print("не изменять портфель")

    # 6. Общая рекомендация
    print(f"\n6. ОБЩАЯ РЕКОМЕНДАЦИЯ:")
    if optimal_income > 0 and optimal_income > passive_income:
        print("   ✓ Рекомендуется активное управление с учетом комиссий")
    elif passive_income > 0:
        print("   ✓ Рекомендуется пассивная стратегия (оставить как есть)")
    else:
        print("   ⚠ Обе стратегии показывают отрицательную доходность")
        print("   Рассмотрите альтернативные инвестиции")


# ------------------------------
# 5. ГЛАВНАЯ ФУНКЦИЯ
# ------------------------------
def main():
    print("=" * 70)
    print("ОПТИМАЛЬНОЕ УПРАВЛЕНИЕ ИНВЕСТИЦИОННЫМ ПОРТФЕЛЕМ")
    print("С учетом комиссий брокеров")
    print("=" * 70)
    print(f"Комиссии: ЦБ1={COM_S1 * 100}%, ЦБ2={COM_S2 * 100}%, Депозиты={COM_D * 100}%")
    print("=" * 70)

    # Запуск алгоритма
    print("\nЗапуск обратной прогонки с учетом комиссий...")
    F, U, _ = backward_induction_with_commissions()

    # Анализ результатов
    analyze_results(F, U)

    # Вывод заключения
    print("\n" + "=" * 70)
    print("ЗАКЛЮЧЕНИЕ")
    print("=" * 70)
    print("1. Учет комиссий существенно влияет на эффективность управления")
    print("2. Высокие комиссии могут сделать активное управление невыгодным")
    print("3. Минимальные остатки ограничивают возможности ребалансировки")
    print("4. При отрицательных ожидаемых доходностях лучше сокращать риски")
    print("=" * 70)


if __name__ == "__main__":

    main()
