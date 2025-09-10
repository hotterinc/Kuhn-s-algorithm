# kuhn_console.py
"""
Консольная реализация алгоритма Куна (DFS) для двудольного графа.
Формат входного файла:
  1-я строка: N (число вершин, нумерация 0..N-1)
  далее N строк: матрица; '-' = нет ребра, любое другое значение (включая 0, отриц.) = есть ребро.
Все графы считаются НЕОРИЕНТИРОВАННЫМИ. Самопетли игнорируются.

Программа:
- читает файл и строит неориентированный граф,
- проверяет двудольность и вычисляет разбиение вершин на доли L/R,
- для каждой вершины u ∈ L запускает dfs(u) (алг. Куна), печатая пошаговые пояснения,
- по окончании пишет ВСЕ шаги в файл (по умолчанию kuhn_steps.log).

Без сторонних библиотек для паросочетаний.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from typing import Dict, List, Set, Tuple


# ---------- Утилиты логирования ----------

class Logger:
    """Копит строки логов и дублирует их в stdout."""
    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, msg: str = "") -> None:
        self.lines.append(msg)
        print(msg)

    def dump_to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


# ---------- Парсинг графа ----------

def load_graph(path: str, logger: Logger) -> List[Set[int]]:
    """
    Загружает граф из файла и возвращает неориентированный список смежности (множества).
    Правило: ребро (i, j) существует, если matrix[i][j] != '-' ИЛИ matrix[j][i] != '-'.
    Самопетли игнорируются.
    """
    logger.log(f"Чтение файла: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        raise ValueError("Файл пуст")

    if not lines[0].isdigit():
        raise ValueError("Первая строка должна содержать число вершин (целое)")

    n = int(lines[0])
    if n <= 0:
        raise ValueError("Число вершин должно быть положительным")

    if len(lines) - 1 < n:
        raise ValueError(f"Ожидается {n} строк матрицы, получено {len(lines)-1}")

    # читаем N строк матрицы
    matrix: List[List[str]] = []
    for i in range(n):
        row = lines[1 + i].split()
        if len(row) != n:
            raise ValueError(f"Строка {i+2} должна содержать {n} элементов")
        matrix.append(row)

    # строим UNDIRECTED список смежности (множества для защиты от дублей)
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != "-" or matrix[j][i] != "-":
                if i != j:
                    adj[i].add(j)
                    adj[j].add(i)

    # краткая сводка
    m = sum(len(s) for s in adj) // 2
    logger.log(f"Загружено: N={n}, |E|={m} (неориентированный)")
    return adj


# ---------- Проверка двудольности ----------

def bipartition(adj: List[Set[int]], logger: Logger) -> Tuple[Set[int], Set[int]]:
    """
    BFS-окраска: возвращает (L, R). Если граф не двудольный — кидает ValueError.
    Изолированные вершины помещаются в левую долю (это не влияет на корректность).
    """
    n = len(adj)
    color: Dict[int, int] = {}
    L: Set[int] = set()
    R: Set[int] = set()

    for start in range(n):
        if start in color:
            continue
        color[start] = 0
        L.add(start)
        q: deque[int] = deque([start])

        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    q.append(v)
                    if color[v] == 0:
                        L.add(v)
                    else:
                        R.add(v)
                elif color[v] == color[u]:
                    # нашли нечётный цикл
                    raise ValueError("Граф не является двудольным")

    # Все изолированные останутся в L (они уже там)
    logger.log(f"Двудольное разбиение: |L|={len(L)}, |R|={len(R)}")
    logger.log(f"  L = {sorted(L)}")
    logger.log(f"  R = {sorted(R)}")
    return L, R


# ---------- Алгоритм Куна (DFS) ----------

def kuhn_maximum_matching(
    adj: List[Set[int]],
    L: Set[int],
    R: Set[int],
    logger: Logger,
) -> Dict[int, int]:
    """
    Каноническая реализация Куна.
    matching: right -> left (ключ — вершина из R, значение — вершина из L).
    Пишет в лог пошаговые пояснения.
    """
    matching: Dict[int, int] = {}

    left_order = sorted(L)
    logger.log("\n=== Запуск алгоритма Куна (DFS) ===")
    logger.log(f"Порядок обхода левых вершин: {left_order}\n")

    def dfs(u: int, visited_right: Set[int], depth: int) -> bool:
        indent = "  " * depth
        for v in sorted(adj[u]):
            if v not in R:
                continue  # пропускаем рёбра, не ведущие в правую долю (на случай "грязных" входов)
            if v in visited_right:
                continue
            visited_right.add(v)

            logger.log(f"{indent}Пробуем ребро {u} — {v}")

            if v not in matching:
                matching[v] = u
                logger.log(f"{indent}  ✔ {v} свободна: ставим пару ({u} — {v})")
                return True
            else:
                prev_u = matching[v]
                logger.log(f"{indent}  {v} занята парой ({prev_u} — {v}), пытаемся переназначить {prev_u}...")
                if dfs(prev_u, visited_right, depth + 1):
                    matching[v] = u
                    logger.log(f"{indent}  ✔ удалось переназначить: теперь ({u} — {v})")
                    return True
                else:
                    logger.log(f"{indent}  ✖ не получилось переназначить {prev_u} через {v}")
        return False

    # Основной цикл: для каждой левой вершины пробуем найти аугментацию
    for idx, u in enumerate(left_order, start=1):
        if u in matching.values():
            logger.log(f"[Шаг {idx}/{len(left_order)}] Левая {u} уже насыщена — пропускаем")
            continue

        logger.log(f"[Шаг {idx}/{len(left_order)}] Пытаемся насытить левую вершину {u}")
        visited: Set[int] = set()
        if dfs(u, visited, depth=1):
            logger.log(f"=> Успех. Размер паросочетания: {len(matching)}\n")
        else:
            logger.log(f"=> Нет аугментирующего пути для {u}. Размер прежний: {len(matching)}\n")

    logger.log("=== Завершение Куна ===\n")
    return matching


# ---------- Главная программа ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Алгоритм Куна (DFS) для двудольного графа: пошаговые пояснения + лог в файл."
    )
    parser.add_argument("input", help="путь к входному файлу с графом")
    parser.add_argument(
        "-o",
        "--output",
        default="kuhn_steps.log",
        help="файл для записи всех шагов (по умолчанию kuhn_steps.log)",
    )
    args = parser.parse_args()

    logger = Logger()
    try:
        adj = load_graph(args.input, logger)
        L, R = bipartition(adj, logger)
    except Exception as e:
        logger.log(f"\nОШИБКА: {e}")
        logger.dump_to_file(args.output)
        sys.exit(1)

    matching = kuhn_maximum_matching(adj, L, R, logger)

    # Итоговая сводка
    k = len(matching)
    Lsz, Rsz = len(L), len(R)
    complete_from_smaller = (k == min(Lsz, Rsz))
    perfect = (Lsz == Rsz == k)

    logger.log("ИТОГ:")
    logger.log(f"  Размер паросочетания |M| = {k}")
    logger.log(f"  |L| = {Lsz}, |R| = {Rsz}")
    logger.log(f"  Полное по меньшей доле: {'ДА' if complete_from_smaller else 'НЕТ'}")
    logger.log(f"  Совершенное (идеальное): {'ДА' if perfect else 'НЕТ'}")
    logger.log("  Пары (left - right), отсортированные по правой вершине:")
    for r, l in sorted(matching.items()):
        logger.log(f"    {l} - {r}")

    # Записываем все шаги в файл
    logger.dump_to_file(args.output)
    logger.log(f"\nПолный лог шагов сохранён в файл: {args.output}")


if __name__ == "__main__":
    main()
