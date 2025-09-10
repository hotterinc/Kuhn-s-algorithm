"""
GUI для пошаговой визуализации алгоритма Куна (полного паросочетание)
в двудольном неориентированном графе (каноническая DFS-реализация).


Формат входа:
  N
  N строк матрицы; '-' — нет ребра, любое иное (включая 0, отриц.) — есть ребро.
Самопетли игнорируются. Без готовых библиотек для паросочетаний.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox, scrolledtext, ttk

matplotlib.use("TkAgg")


class GraphApp:
    """Главное приложение: загрузка, проверка двудольности, Кун-DFS, визуализация и тесты."""

    # ------------------------- Инициализация и UI -------------------------

    def __init__(self, root: tk.Tk) -> None:
        self.root = root #инициализация окна
        self.root.title("Алгоритм Куна для полного паросочетания (DFS)")
        self.root.geometry("1200x800")

        # Состояние графа/алгоритма
        self.graph: Dict[str, List[str]] = {}
        self.left_partition: Set[str] = set()
        self.right_partition: Set[str] = set()

        # matching: right -> left
        self.matching: Dict[str, str] = {}

        # Управление шагами
        self.left_order: List[str] = []
        self.step_index: int = 0  # индекс текущей левой вершины в left_order
        self.auto_running: bool = False
        self.delay = tk.IntVar(value=500)

        # Переменные визуализации
        self.visualize = tk.BooleanVar(value=True)
        self.trace_edges: List[Tuple[str, str]] = []  # попытки на последнем шаге

        # Тестовые кейсы и ожидания
        self.tests = self._make_tests_data()
        self.test_expectations = self._make_test_expectations()
        self._last_test_name: Optional[str] = None

        self.vertex_count: int = 0

        self._build_ui()

    def _build_ui(self) -> None:
        """Интерфейс: две строки кнопок, панед-сплиттеры, поля логов/результатов."""
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_pane)
        right_frame = tk.Frame(main_pane, width=360)
        main_pane.add(left_frame, weight=3)
        main_pane.add(right_frame, weight=1)

        # ----- Панель кнопок (ДВА РЯДА) -----
        btns_container = tk.Frame(left_frame)
        btns_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 6))

        # Ряд 1
        row1 = tk.Frame(btns_container)
        row1.pack(side=tk.TOP, fill=tk.X)

        self.load_btn = tk.Button(row1, text="Загрузить граф из файла", command=self.load_graph)
        self.load_btn.pack(side=tk.LEFT, padx=5, pady=2)

        tests_mb = tk.Menubutton(row1, text="Тесты", relief=tk.RAISED)
        tests_menu = tk.Menu(tests_mb, tearoff=False)
        tests_mb.configure(menu=tests_menu)
        tests_mb.pack(side=tk.LEFT, padx=5, pady=2)
        for name in sorted(self.tests.keys()):
            tests_menu.add_command(label=name, command=lambda n=name: self.run_test(n))
        tests_menu.add_separator()
        tests_menu.add_command(label="Очистить граф", command=self.clear_graph)

        self.find_btn = tk.Button(
            row1, text="Найти полное паросочетание",
            command=self.prepare_algorithm, state=tk.DISABLED
        )
        self.find_btn.pack(side=tk.LEFT, padx=5, pady=2)

        self.step_btn = tk.Button(
            row1, text="Следующий шаг",
            command=lambda: self.next_step(from_auto=False), state=tk.DISABLED
        )
        self.step_btn.pack(side=tk.LEFT, padx=5, pady=2)

        self.auto_btn = tk.Button(
            row1, text="Автозапуск",
            command=self.toggle_auto_run, state=tk.DISABLED
        )
        self.auto_btn.pack(side=tk.LEFT, padx=5, pady=2)

        # Ряд 2
        row2 = tk.Frame(btns_container)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        self.viz_cb = tk.Checkbutton(row2, text="Визуализация", variable=self.visualize)
        self.viz_cb.pack(side=tk.LEFT, padx=5, pady=2)

        speed_frame = tk.Frame(row2)
        speed_frame.pack(side=tk.LEFT, padx=10, pady=2)
        tk.Label(speed_frame, text="Скорость:").pack(side=tk.LEFT)
        self.speed_scale = tk.Scale(
            speed_frame, from_=50, to=2000, orient=tk.HORIZONTAL,
            variable=self.delay, length=160, showvalue=True
        )
        self.speed_scale.pack(side=tk.LEFT)
        tk.Label(speed_frame, text="мс").pack(side=tk.LEFT)

        # Площадка графа
        graph_holder = tk.Frame(left_frame, borderwidth=1, relief=tk.GROOVE)
        graph_holder.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_holder)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        # Правая панель
        status_box = tk.Frame(right_frame)
        status_box.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        tk.Label(status_box, text="Логи выполнения:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        right_pane = ttk.Panedwindow(right_frame, orient=tk.VERTICAL)
        right_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        logs_frame = tk.Frame(right_pane)
        results_frame = tk.Frame(right_pane)
        right_pane.add(logs_frame, weight=3)
        right_pane.add(results_frame, weight=2)

        self.log_text = scrolledtext.ScrolledText(logs_frame, height=20, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        info_frame = tk.Frame(right_frame)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(info_frame, text="Статус:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Граф не загружен")
        tk.Label(info_frame, textvariable=self.status_var, foreground="blue") \
            .grid(row=0, column=1, sticky="w", padx=6)

        tk.Label(info_frame, text="Шаг:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.step_var = tk.StringVar(value="0/0")
        tk.Label(info_frame, textvariable=self.step_var).grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))

        tk.Label(results_frame, text="Результаты:", font=("Arial", 9, "bold")) \
            .pack(anchor=tk.W, pady=(0, 4))
        self.result_text = scrolledtext.ScrolledText(results_frame, height=8, width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------- Алгоритм Куна (DFS) -------------------------

    def prepare_algorithm(self) -> None:
        """Сброс и подготовка шаго-логики: по одной левой вершине за шаг."""
        if not self.graph:
            messagebox.showwarning("Предупреждение", "Сначала загрузите граф!")
            return

        self.add_log("=" * 50)
        self.add_log("ПОДГОТОВКА АЛГОРИТМА КУНА (DFS)")
        self.add_log("=" * 50)

        # Сброс
        self.matching.clear()
        self.step_index = 0
        self.trace_edges = []

        # Порядок левых вершин (все левые — алгоритм сам перераспределит сматченных)
        self.left_order = sorted(self.left_partition, key=int)

        self.step_btn.config(state=tk.NORMAL)
        self.auto_btn.config(state=tk.NORMAL)
        self.update_status("Алгоритм подготовлен. Готов к выполнению.")
        self.update_step_counter()
        self.viz_draw()

    def dfs_augment(self, u: str, visited_right: Set[str]) -> bool:
        """Классический DFS Куна: пытается насытить левую вершину u.
        visited_right — множество уже посещённых правых в текущей попытке.
        """
        for v in sorted(self.graph.get(u, []), key=int):
            if v not in self.right_partition:
                continue  # использовать только рёбра в правую долю

            if v in visited_right:
                continue
            visited_right.add(v)

            # Для визуального трейсинга (оранжевые пунктирные рёбра на шаге)
            self.trace_edges.append((u, v))

            # Если правая свободна — матчим
            if v not in self.matching:
                self.matching[v] = u
                return True

            # Иначе попробуем "сдвинуть" её текущего партнёра
            prev_u = self.matching[v]
            if self.dfs_augment(prev_u, visited_right):
                self.matching[v] = u
                return True

        return False

    def next_step(self, from_auto: bool = False) -> None:
        """Один шаг: берём очередную левую вершину, запускаем dfs(u)."""
        if not from_auto and self.auto_running:
            self.auto_btn.config(text="Автозапуск")
            self.auto_running = False
            return

        if self.step_index >= len(self.left_order):
            self.finalize_algorithm()
            return

        u = self.left_order[self.step_index]
        self.step_index += 1
        self.update_step_counter()

        # Если u уже насыщена предыдущими аугментациями — просто логируем
        if u in self.matching.values():
            self.add_log(f"Шаг {self.step_index}: вершина {u} уже насыщена, пропускаем")
            self.trace_edges = []
            self.viz_draw()
        else:
            self.add_log(f"Шаг {self.step_index}: пытаемся насытить левую вершину {u}")
            visited_right: Set[str] = set()
            self.trace_edges = []
            ok = self.dfs_augment(u, visited_right)
            if ok:
                self.add_log(f"  ✔ Успех: размер паросочетания = {len(self.matching)}")
            else:
                self.add_log(f"  ✖ Нет аугментирующего пути для {u}")
            self.viz_draw()

        if self.step_index >= len(self.left_order):
            self.finalize_algorithm()

    # ------------------------- Результат / отчёт -------------------------

    def finalize_algorithm(self) -> None:
        """Подсчёт и вывод результатов: размер, «полное из меньшей доли», совершенство."""
        self.step_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)

        L = len(self.left_partition)
        R = len(self.right_partition)
        k = len(self.matching)

        complete_from_smaller = (k == min(L, R))
        perfect = (L == R) and (k == L)

        self.add_log("=" * 50)
        self.add_log("АЛГОРИТМ ЗАВЕРШЁН")
        self.add_log(f"Размер паросочетания: {k}")
        self.add_log(f"Полное из меньшей доли: {'ДА' if complete_from_smaller else 'НЕТ'}")
        self.add_log(f"Совершенное (идеальное): {'ДА' if perfect else 'НЕТ'}")
        self.add_log("=" * 50)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Размер паросочетания: {k}\n")
        self.result_text.insert(tk.END, f"|L|={L}, |R|={R}\n")
        self.result_text.insert(
            tk.END, f"Полное из меньшей доли: {'ДА' if complete_from_smaller else 'НЕТ'}\n"
        )
        self.result_text.insert(
            tk.END, f"Совершенное (идеальное): {'ДА' if perfect else 'НЕТ'}\n\n"
        )
        self.result_text.insert(tk.END, "Рёбра паросочетания (left - right):\n")
        for right, left in sorted(self.matching.items(), key=lambda x: (int(x[0]), int(x[1]))):
            self.result_text.insert(tk.END, f"{left} - {right}\n")

        self.update_status("Алгоритм завершен")

        # Если это был встроенный тест — покажем ожидания
        if self._last_test_name and self._last_test_name in self.test_expectations:
            exp = self.test_expectations[self._last_test_name]
            if exp.get("bipartite", True):
                self.add_log(
                    f"[Ожидаемо] размер: {exp['size']}, "
                    f"полное: {'ДА' if exp['complete'] else 'НЕТ'}, "
                    f"совершенное: {'ДА' if exp['perfect'] else 'НЕТ'}"
                )

    # ------------------------- Автозапуск -------------------------

    def toggle_auto_run(self) -> None:
        if not self.auto_running:
            if not self.left_order:
                return
            self.auto_btn.config(text="Остановить")
            self.auto_running = True
            self._auto_tick()
        else:
            self.auto_btn.config(text="Автозапуск")
            self.auto_running = False

    def _auto_tick(self) -> None:
        if self.auto_running and self.step_index < len(self.left_order):
            self.next_step(from_auto=True)
            if self.auto_running:
                self.root.after(self.delay.get(), self._auto_tick)
        else:
            self.auto_btn.config(text="Автозапуск")
            self.auto_running = False

    # ------------------------- Вспомогательные: UI -------------------------

    def viz_draw(
        self,
        highlight_vertices: Optional[Set[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Перерисовать граф (с учетом текущего matching и трассировки шага)."""
        if not self.visualize.get():
            return
        self.draw_graph(
            highlight_vertices=highlight_vertices,
            highlight_edges=highlight_edges if highlight_edges is not None else self.trace_edges,
        )
        self.root.update_idletasks()

    def add_log(self, message: str) -> None:
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        if self.visualize.get():
            self.root.update_idletasks()

    def update_status(self, message: str) -> None:
        self.status_var.set(message)
        if self.visualize.get():
            self.root.update_idletasks()

    def update_step_counter(self) -> None:
        total = len(self.left_order)
        current = min(self.step_index, total)
        self.step_var.set(f"{current}/{total}")

    def clear_graph(self) -> None:
        self.graph.clear()
        self.left_partition.clear()
        self.right_partition.clear()
        self.matching.clear()
        self.vertex_count = 0
        self.left_order = []
        self.step_index = 0
        self.trace_edges = []
        self.log_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.update_status("Граф очищен")
        self.step_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)
        self.find_btn.config(state=tk.DISABLED)
        self.viz_draw()

    # ------------------------- Отрисовка -------------------------

    def draw_graph(
        self,
        highlight_vertices: Optional[Set[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.ax.clear()

        if not self.graph:
            self.ax.text(0.5, 0.5, "Граф не загружен", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return

        highlight_vertices = set(highlight_vertices or [])
        highlight_edges = set(highlight_edges or [])

        # Раскладка: левая доля x=0, правая x=3
        pos: Dict[str, Tuple[int, int]] = {}
        for i, node in enumerate(sorted(self.left_partition, key=int)):
            pos[node] = (0, i)
        for i, node in enumerate(sorted(self.right_partition, key=int)):
            pos[node] = (3, i)

        # Все рёбра (серые)
        for u, nbrs in self.graph.items():
            for v in nbrs:
                if u in pos and v in pos and int(u) < int(v):
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    self.ax.plot([x1, x2], [y1, y2], "gray", alpha=0.45)

        # Рёбра матчинга (красные)
        for right, left in self.matching.items():
            if left in pos and right in pos:
                x1, y1 = pos[left]
                x2, y2 = pos[right]
                self.ax.plot([x1, x2], [y1, y2], "red", linewidth=3)

        # Подсветка попыток шага (оранжевые пунктир)
        for u, v in highlight_edges:
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                self.ax.plot([x1, x2], [y1, y2], "orange", linewidth=2.5, linestyle="dashed")

        # Вершины
        for node in self.left_partition:
            if node in pos and node not in highlight_vertices:
                x, y = pos[node]
                self.ax.plot(x, y, "o", markersize=15, markerfacecolor="lightblue",
                             markeredgecolor="black", markeredgewidth=1)
                self.ax.text(x, y, node, ha="center", va="center")

        for node in self.right_partition:
            if node in pos and node not in highlight_vertices:
                x, y = pos[node]
                self.ax.plot(x, y, "o", markersize=15, markerfacecolor="lightgreen",
                             markeredgecolor="black", markeredgewidth=1)
                self.ax.text(x, y, node, ha="center", va="center")

        for node in highlight_vertices:
            if node in pos:
                x, y = pos[node]
                color = "blue" if node in self.left_partition else "green"
                self.ax.plot(x, y, "o", markersize=20, markerfacecolor=color,
                             markeredgecolor="black", markeredgewidth=2)
                self.ax.text(x, y, node, ha="center", va="center", color="white", weight="bold")

        self.ax.set_xlim(-1, 4)
        self.ax.set_ylim(-1, max(len(self.left_partition), len(self.right_partition)) + 1)
        self.ax.set_axis_off()
        self.ax.set_title("Двудольный граф — алгоритм Куна (DFS)")
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------- Загрузка / парсинг -------------------------

    def load_graph(self) -> None:
        filename = filedialog.askopenfilename(
            title="Выберите файл с графом",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
        )
        if not filename:
            return

        try:
            self.add_log(f"Загрузка графа из файла: {filename}")
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            self._load_graph_from_text(text, source_name=filename)
        except Exception as exc:
            self.add_log(f"ОШИБКА: {exc}")
            messagebox.showerror("Ошибка", f"Ошибка при загрузке файла:\n{exc}")

    def _load_graph_from_text(self, text: str, source_name: str = "текст") -> None:
        vertex_count, matrix = self._parse_graph_text(text)
        self._apply_matrix(matrix, vertex_count, source_name)

    def _parse_graph_text(self, text: str) -> Tuple[int, List[List[str]]]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("Пустой ввод")
        if not lines[0].isdigit():
            raise ValueError("Первая строка должна содержать число вершин")
        n = int(lines[0])
        if n <= 0:
            raise ValueError("Число вершин должно быть положительным")
        if len(lines) - 1 < n:
            raise ValueError(f"Ожидается {n} строк матрицы")

        matrix: List[List[str]] = []
        for i in range(n):
            row = lines[1 + i].split()
            if len(row) != n:
                raise ValueError(f"Строка {i + 2} должна содержать {n} элементов")
            matrix.append(row)
        return n, matrix

    def _apply_matrix(self, matrix: List[List[str]], n: int, source_name: str = "текст") -> None:
        # Словарь смежности
        self.graph = {str(i): [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # самопетли игнорируем
                if matrix[i][j] != "-":
                    u, v = str(i), str(j)
                    self.graph[u].append(v)

        self.vertex_count = n

        # Проверка двудольности
        self.add_log(f"Проверка двудольности графа ({source_name})...")
        if not self.is_bipartite():
            self.add_log("ОШИБКА: Граф не является двудольным!")
            messagebox.showerror("Ошибка", "Граф не является двудольным!")
            # Откат
            self.graph.clear()
            self.left_partition.clear()
            self.right_partition.clear()
            self.viz_draw()
            self.update_status("Граф не загружен")
            self.find_btn.config(state=tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED)
            self.auto_btn.config(state=tk.DISABLED)
            self.update_step_counter()
            return

        # Сброс алгоритма
        self.matching.clear()
        self.left_order = []
        self.step_index = 0
        self.trace_edges = []

        self.add_log(f"Граф успешно загружен из: {source_name}")
        self.add_log(f"Левая доля: {sorted(self.left_partition, key=int)}")
        self.add_log(f"Правая доля: {sorted(self.right_partition, key=int)}")

        self.viz_draw()
        self.update_status("Граф загружен. Нажмите «Найти полное паросочетание».")
        self.find_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)
        self.update_step_counter()

    # ------------------------- Двудольность -------------------------

    def is_bipartite(self) -> bool:
        """BFS-окраска с заполнением левой/правой долей."""
        if not self.graph:
            return False

        color: Dict[str, int] = {}
        self.left_partition.clear()
        self.right_partition.clear()

        for start in range(self.vertex_count):
            s = str(start)
            if s in color:
                continue
            color[s] = 0
            q: deque[str] = deque([s])
            self.left_partition.add(s)

            while q:
                u = q.popleft()
                for v in self.graph.get(u, []):
                    if v not in color:
                        color[v] = 1 - color[u]
                        q.append(v)
                        if color[v] == 0:
                            self.left_partition.add(v)
                        else:
                            self.right_partition.add(v)
                    elif color[v] == color[u]:
                        return False

        # Изолированные вершины — в левую (не влияет на корректность)
        for i in range(self.vertex_count):
            si = str(i)
            if si not in self.left_partition and si not in self.right_partition:
                self.left_partition.add(si)

        return True

    # ------------------------- Тесты -------------------------

    def _make_tests_data(self) -> Dict[str, str]:
        return {
            "1) perfect_6": """6
- - - 1 1 -
- - - 1 1 -
- - - 1 1 1
1 1 1 - - -
1 1 1 - - -
- - 1 - - -""",
            "2) star_5": """5
- 1 1 1 1
1 - - - -
1 - - - -
1 - - - -
1 - - - -""",
            "3) hall_violation_8": """8
- - - - 1 1 - -
- - - - 1 1 - -
- - - - 1 1 - -
- - - - - - 1 -
1 1 1 - - - - -
1 1 1 - - - - -
- - - 1 - - - -
- - - - - - - -""",
            "4) empty_6": """6
- - - - - -
- - - - - -
- - - - - -
- - - - - -
- - - - - -
- - - - - -""",
            "5) not_bipartite_triangle_5": """5
- 1 1 - -
1 - 1 - -
1 1 - - -
- - - - 1
- - - 1 -""",
            "6) two_edges_isolates_7": """7
- - - - - 1 -
- - - - - - 1
- - - - - - -
- - - - - - -
- - - - - - -
1 - - - - - -
- 1 - - - - -""",
            "7) K5,5_complete_10": """10
- - - - - 1 1 1 1 1
- - - - - 1 1 1 1 1
- - - - - 1 1 1 1 1
- - - - - 1 1 1 1 1
- - - - - 1 1 1 1 1
1 1 1 1 1 - - - - -
1 1 1 1 1 - - - - -
1 1 1 1 1 - - - - -
1 1 1 1 1 - - - - -
1 1 1 1 1 - - - - -""",
            "8) weighted_perfect_6": """6
- - - 0 -5 -
- - - -1 2 -
- - - 0 - 7
0 -1 0 - - -
-5 2 - - - -
- - 7 - - -""",
        }

    def _make_test_expectations(self) -> Dict[str, Dict]:
        return {
            "1) perfect_6": dict(size=3, complete=True, perfect=True, bipartite=True),
            "2) star_5": dict(size=1, complete=True, perfect=False, bipartite=True),
            "3) hall_violation_8": dict(size=3, complete=False, perfect=False, bipartite=True),
            "4) empty_6": dict(size=0, complete=True, perfect=False, bipartite=True),
            "5) not_bipartite_triangle_5": dict(bipartite=False),
            "6) two_edges_isolates_7": dict(size=2, complete=True, perfect=False, bipartite=True),
            "7) K5,5_complete_10": dict(size=5, complete=True, perfect=True, bipartite=True),
            "8) weighted_perfect_6": dict(size=3, complete=True, perfect=True, bipartite=True),
        }

    def run_test(self, name: str) -> None:
        self._last_test_name = name
        text = self.tests[name]
        self.log_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)

        self.add_log(f"=== Тест: {name} ===")
        exp = self.test_expectations.get(name, {})
        if exp.get("bipartite", True):
            self.add_log(
                f"[Ожидаемо] размер={exp['size']}, "
                f"полное={'ДА' if exp['complete'] else 'НЕТ'}, "
                f"совершенное={'ДА' if exp['perfect'] else 'НЕТ'}"
            )
        else:
            self.add_log("[Ожидаемо] граф не двудольный → должна быть ошибка загрузки")

        self._load_graph_from_text(text, source_name=f"тест «{name}»")


# ------------------------- Точка входа -------------------------

def main() -> None:
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
