import json
import numpy as np


class LinearProgrammingSolver:
    def __init__(self):
        self.A = None
        self.b = None
        self.c = None
        self.basis = None
        self.non_basis = None
        self.artificial_vars = []
        self.problem_type = None
        self.num_vars = 0
        self.num_constraints = 0
        self.constraint_types = []
        self.variable_bounds = []  # Границы переменных
        self.substitutions = {}  # Словарь замен переменных

    def read_problem(self, filename):
        """Считывание задачи из JSON файла"""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.problem_type = data['objective']
        self.c = np.array(data['c'], dtype=float)
        self.num_vars = len(self.c)

        # Собираем ограничения
        constraints = data['constraints']
        self.num_constraints = len(constraints)
        self.constraint_types = [constraint['type'] for constraint in constraints]

        # Инициализируем матрицы
        A_list = []
        b_list = []

        for constraint in constraints:
            A_list.append(constraint['coefficients'])
            b_list.append(constraint['value'])

        self.A = np.array(A_list, dtype=float)
        self.b = np.array(b_list, dtype=float)

        # Читаем границы переменных (если есть)
        self.variable_bounds = data.get('variable_bounds', [])
        if not self.variable_bounds:
            # По умолчанию все переменные >= 0
            self.variable_bounds = [{"lower": 0, "upper": None} for _ in range(self.num_vars)]

        # Заменяем None на ±inf
        for bounds in self.variable_bounds:
            if bounds.get('lower') is None:
                bounds['lower'] = -np.inf
            if bounds.get('upper') is None:
                bounds['upper'] = np.inf

        print("Исходная задача:")
        print(f"Целевая функция: {self.problem_type} Z = {self.c}")
        print("Ограничения:")
        for i, constraint in enumerate(constraints):
            print(f"{constraint['coefficients']} {constraint['type']} {constraint['value']}")
        print("Границы переменных:")
        for i, bounds in enumerate(self.variable_bounds):
            lower = bounds.get('lower', '-∞')
            upper = bounds.get('upper', '∞')
            print(f"x{i + 1}: {lower} ≤ x{i + 1} ≤ {upper}")
        print()

    def handle_variable_bounds(self):
        """Обработка границ переменных"""
        print("Обработка границ переменных...")

        new_vars_count = 0
        self.substitutions = {}  # Словарь замен: исходная_переменная -> [положительная, отрицательная]

        # Проходим по всем переменным и обрабатываем границы
        for i in range(self.num_vars):
            bounds = self.variable_bounds[i]
            lower = bounds.get('lower', -np.inf)
            upper = bounds.get('upper', np.inf)

            # Если переменная может быть отрицательной, заменяем её
            if lower < 0:
                # Заменяем x на x⁺ - x⁻, где x⁺ ≥ 0, x⁻ ≥ 0
                self.substitutions[i] = [self.num_vars + new_vars_count, self.num_vars + new_vars_count + 1]
                new_vars_count += 2

        if not self.substitutions:
            print("Все переменные неотрицательные, преобразования не нужны")
            return

        # Создаем расширенную матрицу
        new_A = np.zeros((self.num_constraints, self.num_vars + new_vars_count))
        new_c = np.zeros(self.num_vars + new_vars_count)

        # Копируем исходные коэффициенты
        new_A[:, :self.num_vars] = self.A
        new_c[:self.num_vars] = self.c

        # Применяем замены
        for orig_var, (pos_var, neg_var) in self.substitutions.items():
            # Заменяем x на x⁺ - x⁻
            # Коэффициенты для x⁺
            new_A[:, pos_var] = self.A[:, orig_var]
            # Коэффициенты для x⁻ (с минусом)
            new_A[:, neg_var] = -self.A[:, orig_var]

            # Целевая функция
            new_c[pos_var] = self.c[orig_var]  # коэффициент для x⁺
            new_c[neg_var] = -self.c[orig_var]  # коэффициент для x⁻

            # Обнуляем исходную переменную (теперь она не используется)
            new_A[:, orig_var] = 0
            new_c[orig_var] = 0

        self.A = new_A
        self.c = new_c
        self.num_vars += new_vars_count

        print(f"После обработки границ переменных: {self.num_vars} переменных")
        print(f"Выполнено замен: {len(self.substitutions)}")
        for orig, (pos, neg) in self.substitutions.items():
            print(f"  x{orig + 1} = x{pos + 1} - x{neg + 1}")
        print()

    def to_canonical_form(self):
        """Приведение к канонической форме"""
        print("Приведение к канонической форме...")

        # Сначала обрабатываем границы переменных
        self.handle_variable_bounds()

        # Для минимизации меняем знак целевой функции (будем решать как максимизацию)
        if self.problem_type == 'min':
            self.c = -self.c  # min Z -> max (-Z)

        # Определяем количество дополнительных переменных
        slack_vars_count = 0
        for i in range(self.num_constraints):
            if self.constraint_types[i] == '<=':
                slack_vars_count += 1
            elif self.constraint_types[i] == '>=':
                slack_vars_count += 1  # Для >= добавляем surplus переменную

        # Создаем расширенную матрицу A
        new_A = np.zeros((self.num_constraints, self.num_vars + slack_vars_count))
        new_A[:, :self.num_vars] = self.A

        # Создаем расширенный вектор c
        new_c = np.zeros(self.num_vars + slack_vars_count)
        new_c[:self.num_vars] = self.c

        # Добавляем slack и surplus переменные
        slack_idx = self.num_vars
        for i in range(self.num_constraints):
            if self.constraint_types[i] == '<=':
                new_A[i, slack_idx] = 1  # Slack переменная
                slack_idx += 1
            elif self.constraint_types[i] == '>=':
                new_A[i, slack_idx] = -1  # Surplus переменная
                slack_idx += 1

        self.A = new_A
        self.c = new_c
        self.num_vars += slack_vars_count

        print("Каноническая форма:")
        print(f"A = \n{self.A}")
        print(f"b = {self.b}")
        print(f"c = {self.c}")
        print(f"Новое количество переменных: {self.num_vars}")
        print()

    def form_auxiliary_problem(self):
        """Формирование вспомогательной задачи"""
        print("Формирование вспомогательной задачи...")

        # Определяем, для каких ограничений нужны искусственные переменные
        artificial_needed = []

        for i in range(self.num_constraints):
            if self.constraint_types[i] == '=':
                artificial_needed.append(i)
            elif self.constraint_types[i] == '>=':
                artificial_needed.append(i)

        # Добавляем искусственные переменные
        artificial_count = len(artificial_needed)

        if artificial_count == 0:
            print("Искусственные переменные не нужны")
            self.aux_c = np.zeros(self.num_vars)
            return

        # Расширяем матрицу A
        new_A = np.zeros((self.num_constraints, self.num_vars + artificial_count))
        new_A[:, :self.num_vars] = self.A

        # Добавляем искусственные переменные
        for idx, constraint_idx in enumerate(artificial_needed):
            new_A[constraint_idx, self.num_vars + idx] = 1

        self.A = new_A

        # Целевая функция вспомогательной задачи: минимизация суммы искусственных переменных
        self.aux_c = np.zeros(self.num_vars + artificial_count)
        for i in range(artificial_count):
            self.aux_c[self.num_vars + i] = 1

        self.artificial_vars = list(range(self.num_vars, self.num_vars + artificial_count))
        self.num_vars += artificial_count

        print(f"Добавлено {artificial_count} искусственных переменных: {self.artificial_vars}")
        print(f"Матрица A после добавления искусственных переменных:\n{self.A}")
        print(f"Целевая функция вспомогательной задачи: {self.aux_c}")
        print()

    def find_initial_basis(self):
        """Поиск начального базиса"""
        basis = []
        non_basis = list(range(self.num_vars))

        # Сначала пробуем найти единичные столбцы среди исходных и дополнительных переменных
        for j in range(self.num_vars):
            if j in self.artificial_vars:
                continue

            col = self.A[:, j]
            if np.sum(np.abs(col)) == 1 and np.sum(col == 1) == 1:
                # Нашли столбец с одной единицей
                row_idx = np.where(col == 1)[0][0]
                if row_idx not in [np.where(self.A[:, var] == 1)[0][0] for var in basis if
                                   var not in self.artificial_vars]:
                    basis.append(j)
                    non_basis.remove(j)

        # Добавляем искусственные переменные для оставшихся строк
        for i in range(self.num_constraints):
            row_covered = False
            for var in basis:
                if self.A[i, var] == 1:
                    row_covered = True
                    break

            if not row_covered:
                # Ищем искусственную переменную для этой строки
                for j in self.artificial_vars:
                    if self.A[i, j] == 1 and j not in basis:
                        basis.append(j)
                        if j in non_basis:
                            non_basis.remove(j)
                        break

        return basis, non_basis

    def solve_auxiliary_problem(self):
        """Решение вспомогательной задачи"""
        print("Решение вспомогательной задачи...")

        # Находим начальный базис
        basis, non_basis = self.find_initial_basis()

        print(f"Начальный базис: {basis}")
        print(f"Небазисные переменные: {non_basis}")

        # Проверяем, что базис корректен
        B = self.A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
            x_b = B_inv @ self.b
            print(f"Начальное базисное решение: {x_b}")
        except np.linalg.LinAlgError:
            print("Ошибка: вырожденная матрица базиса")
            return {'status': 'infeasible'}

        # Симплекс-метод для вспомогательной задачи
        result = self.simplex_method(self.A, self.b, self.aux_c, basis, non_basis, is_auxiliary=True)

        if result['status'] == 'optimal':
            aux_value = result['value']
            print(f"Значение вспомогательной задачи: {aux_value}")

            if abs(aux_value) > 1e-10:
                print("Вспомогательная задача имеет положительное значение - исходная задача несовместна")
                return {'status': 'infeasible'}
            else:
                print("Вспомогательная задача решена успешно, переходим к основной")
                return {'status': 'feasible', 'basis': result['basis']}
        else:
            return result

    def solve_main_problem(self, initial_basis):
        """Решение основной задачи"""
        print("Решение основной задачи...")

        # Удаляем искусственные переменные из базиса
        basis = [var for var in initial_basis if var not in self.artificial_vars]

        # Удаляем столбцы искусственных переменных
        if self.artificial_vars:
            columns_to_keep = [i for i in range(self.num_vars) if i not in self.artificial_vars]
            self.A = self.A[:, columns_to_keep]
            self.c = self.c[columns_to_keep]

            # Обновляем индексы базиса
            new_basis = []
            for var in basis:
                new_idx = columns_to_keep.index(var)
                new_basis.append(new_idx)
            basis = new_basis

            self.num_vars = len(columns_to_keep)

        non_basis = [i for i in range(self.num_vars) if i not in basis]

        print(f"Базис для основной задачи: {basis}")
        print(f"Небазисные переменные: {non_basis}")
        print(f"Целевая функция основной задачи: {self.c}")

        result = self.simplex_method(self.A, self.b, self.c, basis, non_basis, is_auxiliary=False)

        return result

    def simplex_method(self, A, b, c, basis, non_basis, is_auxiliary=False):
        """Реализация симплекс-метода с учетом границ переменных"""
        max_iterations = 100
        iteration = 0

        print(f"\nНачало симплекс-метода ({'вспомогательная' if is_auxiliary else 'основная'} задача)")

        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Итерация {iteration} ---")

            # Шаг 1: Вычисление базисного решения
            B = A[:, basis]
            try:
                B_inv = np.linalg.inv(B)
                x_b = B_inv @ b
            except np.linalg.LinAlgError:
                return {'status': 'error', 'message': 'Вырожденная матрица базиса'}

            c_b = c[basis]

            print(f"Базисные переменные: {basis}")
            print(f"Базисное решение: {x_b}")

            # Проверка допустимости
            if any(x_b_i < -1e-10 for x_b_i in x_b):
                print("Нарушена допустимость базисного решения")
                return {'status': 'infeasible'}

            # Шаг 2: Вычисление оценок
            improving_var = None

            if is_auxiliary:
                # Для вспомогательной задачи минимизируем
                best_reduced_cost = 0
                for j in non_basis:
                    A_j = A[:, j]
                    y_j = B_inv @ A_j
                    reduced_cost = c[j] - c_b @ y_j

                    if reduced_cost < best_reduced_cost - 1e-10:
                        best_reduced_cost = reduced_cost
                        improving_var = j
            else:
                # Для основной задачи максимизируем
                best_reduced_cost = -1e-10
                for j in non_basis:
                    A_j = A[:, j]
                    y_j = B_inv @ A_j
                    reduced_cost = c[j] - c_b @ y_j

                    if reduced_cost > best_reduced_cost + 1e-10:
                        best_reduced_cost = reduced_cost
                        improving_var = j

            print(f"Лучшая оценка: {best_reduced_cost:.6f}")

            if improving_var is None:
                # Оптимальное решение найдено
                x = np.zeros(A.shape[1])
                x[basis] = x_b

                # Проверяем границы небазисных переменных
                for j in non_basis:
                    # Небазисные переменные должны быть на границах (0 или upper bound)
                    if x[j] < -1e-10 or x[j] > 1e-10:
                        print(f"Предупреждение: небазисная переменная x{j} не на границе: {x[j]}")

                value = c @ x

                print(f"Оптимальное решение найдено!")
                print(f"x = {x}")
                print(f"Значение целевой функции: {value:.6f}")

                return {'status': 'optimal', 'value': value, 'solution': x, 'basis': basis}

            print(f"Вводимая переменная: {improving_var}")

            # Шаг 3: Определение выводимой переменной
            A_k = A[:, improving_var]
            y_k = B_inv @ A_k

            print(f"Направление: {y_k}")

            if all(y_k_i <= 1e-10 for y_k_i in y_k):
                print("Все компоненты направления неположительные - задача неограниченная")
                return {'status': 'unbounded'}

            # Находим минимальное положительное отношение
            min_ratio = float('inf')
            leaving_idx = -1

            for i in range(len(basis)):
                if y_k[i] > 1e-10:
                    ratio = x_b[i] / y_k[i]
                    if ratio < min_ratio - 1e-10:
                        min_ratio = ratio
                        leaving_idx = i

            if leaving_idx == -1:
                print("Не найдено положительных отношений - задача неограниченная")
                return {'status': 'unbounded'}

            leaving_var = basis[leaving_idx]
            print(f"Выводимая переменная: {leaving_var} (индекс {leaving_idx})")
            print(f"Минимальное отношение: {min_ratio:.6f}")

            # Шаг 4: Обновление базиса
            basis[leaving_idx] = improving_var
            non_basis = [j for j in non_basis if j != improving_var]
            non_basis.append(leaving_var)

            print(f"Новый базис: {basis}")

        return {'status': 'max_iterations_exceeded'}

    def solve(self, filename):
        """Основной метод решения"""
        try:
            # Шаг 1: Считывание задачи
            self.read_problem(filename)

            # Шаг 2: Приведение к канонической форме
            self.to_canonical_form()

            # Шаг 3: Формирование вспомогательной задачи
            self.form_auxiliary_problem()

            # Шаг 4: Решение вспомогательной задачи
            aux_result = self.solve_auxiliary_problem()

            if aux_result['status'] != 'feasible':
                return self.format_result(aux_result)

            # Шаг 5: Решение основной задачи
            main_result = self.solve_main_problem(aux_result['basis'])

            return self.format_result(main_result)

        except Exception as e:
            return f"Ошибка при решении: {str(e)}"

    def format_result(self, result):
        """Форматирование результата"""
        print("\n" + "=" * 60)
        print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
        print("=" * 60)

        if result['status'] == 'optimal':
            # Восстанавливаем исходные переменные из преобразованных
            original_solution = self.reconstruct_original_solution(result['solution'])
            value = result['value']

            # Корректируем знак для минимизации
            if self.problem_type == 'min':
                final_value = -value  # Т.к. мы решали max (-Z)
            else:
                final_value = value

            output = f"ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО:\n"
            for i, val in enumerate(original_solution):
                output += f"x{i + 1} = {val:.6f}\n"
            output += f"Оптимальная точка x* = {original_solution}\n"
            output += f"Значение целевой функции: Z = {final_value:.6f}"

        elif result['status'] == 'infeasible':
            output = "ЗАДАЧА НЕ ИМЕЕТ ДОПУСТИМЫХ РЕШЕНИЙ\n"
            output += "Причина: область допустимых решений пуста"

        elif result['status'] == 'unbounded':
            output = "ЦЕЛЕВАЯ ФУНКЦИЯ НЕ ОГРАНИЧЕНА\n"
            output += "Причина: можно бесконечно улучшать значение целевой функции"

        else:
            output = f"ОШИБКА ПРИ РЕШЕНИИ:\n{result.get('message', 'Неизвестная ошибка')}"

        return output

    def reconstruct_original_solution(self, transformed_solution):
        """Восстановление исходного решения из преобразованного"""
        original_solution = np.zeros(len(self.variable_bounds))

        for i in range(len(self.variable_bounds)):
            if i in self.substitutions:
                pos_var, neg_var = self.substitutions[i]
                # x = x⁺ - x⁻
                original_solution[i] = transformed_solution[pos_var] - transformed_solution[neg_var]
            else:
                original_solution[i] = transformed_solution[i]

        return original_solution


# Основная программа
if __name__ == "__main__":
    solver = LinearProgrammingSolver()

    try:
        result = solver.solve("problem.json")
        print(result)

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
