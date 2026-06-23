import numpy as np
import sympy as sp
from sklearn.metrics import r2_score
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Определяем символьные переменные
alpha, beta, del_r = sp.symbols('alpha beta del_r', real=True)

# Твои полиномы в виде символьных выражений
polynomials = {
    'C_L0_1': 6.871e-3 + 5.482e-2 * alpha - 2.979e-18 * beta - 1.486e-4 * alpha ** 2 - 9.3e-20 * alpha * beta - 3.223e-5 * beta ** 2,
    'C_D0_1': 4.389e-4 + 6.723e-5 * alpha + 3.758e-19 * beta + 3.242e-4 * alpha ** 2 - 3.321e-21 * alpha * beta + 3.347e-5 * beta ** 2,
    'C_Y0_1': -5.925e-19 + 4.448e-20 * alpha - 5.552e-3 * beta + 1.827e-20 * alpha ** 2 - 1.846e-6 * alpha * beta,
    'C_L_right_1': 6.538e-3 + 5.15e-2 * alpha + 7.507e-3 * del_r - 1.637e-6 * alpha ** 2 - 8.98e-7 * alpha * del_r - 1.557e-6 * del_r ** 2,
    'C_D_right_1': 8.587e-4 + 7.499e-5 * alpha + 1.662e-5 * del_r + 3.313e-4 * alpha ** 2 + 1.054e-4 * alpha * del_r + 2.64e-5 * del_r ** 2,
    'C_Y_right_1': -1.034e-4 + 3.091e-6 * alpha - 5.312e-6 * del_r + 3.102e-6 * alpha ** 2 - 3.124e-5 * alpha * del_r - 8.718e-6 * del_r ** 2,
    'C_l0_1': -8.194e-20 + 9.856e-20 * alpha + 4.688e-5 * beta + 7.088e-21 * alpha ** 2 - 1.59e-5 * alpha * beta,
    'C_m0_1': 1.669e-2 - 2.928e-3 * alpha + 3.098e-19 * beta - 2.766e-5 * alpha ** 2 + 5.812e-21 * alpha * beta - 8.81e-6 * beta ** 2,
    'C_n0_1': -4.207e-19 - 3.847e-20 * alpha + 6.897e-4 * beta - 3.475e-21 * alpha ** 2 + 5.16e-7 * alpha * beta + 6.613e-23 * alpha ** 3 + 2.06e-6 * alpha ** 2 * beta,
    'C_l_right_1': 1.202e-4 + 5.772e-4 * alpha - 1.936e-3 * del_r - 1.99e-8 * alpha ** 2 + 6.581e-8 * alpha * del_r + 9.733e-8 * del_r ** 2 + 3.013e-7 * alpha ** 2 * del_r + 1.096e-6 * alpha * del_r ** 2 + 1.056e-7 * del_r ** 3,
    'C_m_right_1': 1.617e-2 - 2.303e-3 * alpha - 3.134e-3 * del_r - 1.25e-5 * alpha ** 2 - 2.356e-7 * alpha * del_r + 3.879e-7 * del_r ** 2 + 8.808e-7 * alpha ** 2 * del_r + 3.003e-6 * alpha * del_r ** 2 + 3.427e-7 * del_r ** 3,
    'C_n_right_1': -1.76e-5 + 7.722e-7 * alpha - 1.563e-6 * del_r + 1.336e-6 * alpha ** 2 + 5.525e-7 * alpha * del_r + 9.061e-7 * del_r ** 2,
}


def get_terms(expr):
    """Извлекает все члены полинома в виде списка"""
    if expr.is_Add:
        return list(expr.args)
    else:
        return [expr]


def evaluate_polynomial(expr, alpha_val, beta_val, del_r_val):
    """Вычисляет значение полинома в точке"""
    subs_dict = {alpha: alpha_val, beta: beta_val, del_r: del_r_val}
    try:
        return float(expr.subs(subs_dict).evalf())
    except:
        return 0.0


def generate_test_points(num_points=1000, ranges=None):
    """Генерирует тестовые точки"""
    if ranges is None:
        ranges = {
            alpha: (-10, 10),
            beta: (-10, 10),
            del_r: (-10, 10)
        }

    points = []
    for _ in range(num_points):
        point = {
            alpha: np.random.uniform(*ranges[alpha]),
            beta: np.random.uniform(*ranges[beta]),
            del_r: np.random.uniform(*ranges[del_r])
        }
        points.append(point)
    return points


def compute_r2_for_model(original_expr, terms_to_exclude=None, test_points=None):
    """
    Вычисляет R² для модели с исключенными членами
    """
    if test_points is None:
        test_points = generate_test_points(500)

    all_terms = get_terms(original_expr)

    if terms_to_exclude:
        remaining_terms = [term for i, term in enumerate(all_terms) if i not in terms_to_exclude]
        reduced_expr = sum(remaining_terms) if remaining_terms else sp.Integer(0)
    else:
        reduced_expr = original_expr

    y_true = []
    y_pred = []

    for point in test_points:
        y_true.append(evaluate_polynomial(original_expr, point[alpha], point[beta], point[del_r]))
        y_pred.append(evaluate_polynomial(reduced_expr, point[alpha], point[beta], point[del_r]))

    try:
        return r2_score(y_true, y_pred)
    except:
        return 0.0


def simplify_polynomial(expr, test_points=None, target_r2=0.95):
    """
    Упрощает полином, удаляя наименее важные члены,
    пока R² не упадет ниже target_r2
    """
    if test_points is None:
        test_points = generate_test_points(1000)

    # Получаем все члены
    all_terms = get_terms(expr)
    n_terms = len(all_terms)

    if n_terms <= 1:
        return expr, [], 1.0

    # Вычисляем важность каждого члена
    r2_full = compute_r2_for_model(expr, test_points=test_points)

    # Сортируем члены по важности (от наименее важных к наиболее)
    importance = []
    for i in range(n_terms):
        r2_without = compute_r2_for_model(expr, terms_to_exclude=[i], test_points=test_points)
        delta = r2_full - r2_without
        importance.append((i, delta, all_terms[i]))

    # Сортируем по возрастанию важности (наименее важные сначала)
    importance_sorted = sorted(importance, key=lambda x: x[1])

    # Пробуем удалять наименее важные члены
    terms_to_remove = []
    current_r2 = r2_full

    for i, delta, term in importance_sorted:
        # Пробуем удалить этот член
        test_remove = terms_to_remove + [i]
        new_r2 = compute_r2_for_model(expr, terms_to_exclude=test_remove, test_points=test_points)

        # Если R² все еще выше порога, удаляем член
        if new_r2 >= target_r2:
            terms_to_remove.append(i)
            current_r2 = new_r2
        else:
            # Если удаление ухудшает R² ниже порога, останавливаемся
            break

    # Создаем упрощенное выражение
    remaining_terms = [term for idx, term in enumerate(all_terms) if idx not in terms_to_remove]
    simplified_expr = sum(remaining_terms) if remaining_terms else sp.Integer(0)

    # Сортируем индексы удаленных членов для красивого вывода
    terms_to_remove.sort()

    return simplified_expr, terms_to_remove, current_r2


def analyze_polynomial(name, expr, test_points=None):
    """
    Анализирует полином и упрощает его
    """
    print(f"\n{'=' * 70}")
    print(f"Анализ полинома: {name}")
    print(f"{'=' * 70}")

    terms = get_terms(expr)
    n_terms = len(terms)

    if test_points is None:
        test_points = generate_test_points(1000)

    # Вычисляем R² для полной модели
    r2_full = compute_r2_for_model(expr, test_points=test_points)
    print(f"\nИсходный полином:")
    print(f"  Количество членов: {n_terms}")
    print(f"  R² полной модели: {r2_full:.8f}")

    # Показываем все члены
    print(f"\nВсе члены полинома:")
    for i, term in enumerate(terms):
        term_str = str(term)
        if len(term_str) > 60:
            term_str = term_str[:57] + "..."
        print(f"  {i}: {term_str}")

    # Упрощаем полином с разными порогами
    print(f"\n{'=' * 70}")
    print("Упрощение полинома (целевой R² = 95%):")
    print(f"{'=' * 70}")

    simplified_expr, removed_indices, r2_simplified = simplify_polynomial(
        expr, test_points, target_r2=0.95
    )

    # Показываем результат
    removed_terms = [terms[i] for i in removed_indices]
    remaining_terms = [terms[i] for i in range(n_terms) if i not in removed_indices]

    print(f"\nУДАЛЕННЫЕ члены ({len(removed_indices)} шт.):")
    for i, idx in enumerate(removed_indices):
        print(f"  {i + 1}. {terms[idx]}")

    print(f"\nОСТАВШИЕСЯ члены ({len(remaining_terms)} шт.):")
    for i, term in enumerate(remaining_terms):
        print(f"  {i + 1}. {term}")

    print(f"\nРезультат упрощения:")
    print(f"  R² упрощенной модели: {r2_simplified:.8f}")
    print(f"  Потеря точности: {(r2_full - r2_simplified):.8f} ({(1 - r2_simplified / r2_full) * 100:.2f}%)")
    print(
        f"  Сокращение членов: {n_terms} → {len(remaining_terms)} ({-((n_terms - len(remaining_terms)) / n_terms * 100):.1f}%)")

    print(f"\nУпрощенный полином:")
    print(f"  {name}_simplified = {simplified_expr}")

    return {
        'name': name,
        'original': expr,
        'simplified': simplified_expr,
        'removed_indices': removed_indices,
        'removed_terms': removed_terms,
        'remaining_terms': remaining_terms,
        'r2_full': r2_full,
        'r2_simplified': r2_simplified,
        'n_terms_original': n_terms,
        'n_terms_simplified': len(remaining_terms)
    }


# Основная часть программы
def main():
    print("Анализ влияния членов полиномов на R² и упрощение")
    print("=" * 70)

    # Генерируем тестовые точки
    ranges = {
        alpha: (-5, 5),
        beta: (-5, 5),
        del_r: (-30, 30)
    }
    test_points = generate_test_points(1000, ranges)

    # Анализируем каждый полином
    all_results = {}
    for name, expr in polynomials.items():
        all_results[name] = analyze_polynomial(name, expr, test_points)

    # Выводим сводную таблицу всех упрощенных полиномов
    print(f"\n{'=' * 70}")
    print("СВОДНАЯ ТАБЛИЦА УПРОЩЕННЫХ ПОЛИНОМОВ")
    print(f"{'=' * 70}")

    summary_data = []
    for name, result in all_results.items():
        summary_data.append({
            'Полином': name,
            'Исходных_членов': result['n_terms_original'],
            'Осталось_членов': result['n_terms_simplified'],
            'Удалено_членов': result['n_terms_original'] - result['n_terms_simplified'],
            'R²_полный': result['r2_full'],
            'R²_упрощенный': result['r2_simplified'],
            'Потеря_R²': result['r2_full'] - result['r2_simplified']
        })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    # Сохраняем результаты в CSV
    df.to_csv('polynomial_simplification_summary.csv', index=False, encoding='utf-8-sig')

    # Сохраняем все упрощенные полиномы в файл
    print(f"\n{'=' * 70}")
    print("ВСЕ УПРОЩЕННЫЕ ПОЛИНОМЫ:")
    print(f"{'=' * 70}")

    with open('simplified_polynomials.txt', 'w', encoding='utf-8') as f:
        f.write("УПРОЩЕННЫЕ ПОЛИНОМЫ (R² >= 95%)\n")
        f.write("=" * 70 + "\n\n")

        for name, result in all_results.items():
            print(f"\n{name}_simplified = {result['simplified']}")
            f.write(f"{name}_simplified = {result['simplified']}\n")
            f.write(f"  R² = {result['r2_simplified']:.8f} (было {result['r2_full']:.8f})\n")
            f.write(
                f"  Удалено членов: {result['n_terms_original'] - result['n_terms_simplified']} из {result['n_terms_original']}\n\n")

    print(f"\nРезультаты сохранены в файлы:")
    print(f"  - 'simplified_polynomials.txt' - все упрощенные полиномы")
    print(f"  - 'polynomial_simplification_summary.csv' - сводная таблица")


if __name__ == "__main__":
    main()