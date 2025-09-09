"""
GUI для пошаговой визуализации алгоритма Куна (максимальное паросочетание)
в двудольном неориентированном графе. Граф загружается из текстового файла
(или из встроенных тестов). Формат:
  1-я строка — число вершин N
  далее N строк — матрица смежности/весов; '-' означает отсутствие ребра,
  любое другое значение — наличие ребра. Самопетли игнорируются.

Особенности:
- Пошаговый BFS-поиск увеличивающих путей (Кун).
- Тумблер «Визуализация» (ускоряет, отключая отрисовку).
- Меню «Тесты» с 8 готовыми кейсами и ожиданиями.
- Резайзабельные панели (панед-сплиттеры).
- Кнопки в ДВА ряда: верхний ряд действий, нижний — опции и скорость.

Требование задачи: алгоритмы на графах реализованы вручную,
без готовых библиотек для паросочетаний.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox, scrolledtext, ttk

matplotlib.use("TkAgg")


class GraphApp:
    """Главное приложение: загрузка, проверка двудольности, Кун, визуализация и тесты."""

    # ------------------------- Инициализация и UI -------------------------

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Алгоритм Куна для полного паросочетания")
        self.root.geometry("1200x800")

        # Состояние графа/алгоритма
        self.graph: Dict[str, List[str]] = {}
        self.left_partition: Set[str] = set()
        self.right_partition: Set[str] = set()
        self.matching: Dict[str, str] = {}  # отображение right -> left
        self.current_step: int = 0
        self.steps: List = []
        self.delay = tk.IntVar(value=500)
        self.vertex_count: int = 0
        self.auto_running: bool = False
        self.bfs_queue: deque[str] = deque()
        self.visited: Set[str] = set()
        self.parent: Dict[str, Optional[str]] = {}
        self.left_vertices: List[str] = []
        self.current_vertex_index: int = 0
        self.current_start: Optional[str] = None

        # Тумблер визуализации
        self.visualize = tk.BooleanVar(value=True)

        # Тестовые кейсы и ожидания
        self.tests = self._make_tests_data()
        self.test_expectations = self._make_test_expectations()
        self._last_test_name: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Создаёт весь интерфейс: две строки кнопок, панед-сплиттеры, поля логов/результатов."""
        # Главный горизонтальный сплиттер: слева — граф/кнопки, справа — логи/результаты
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_pane)
        right_frame = tk.Frame(main_pane, width=360)
        main_pane.add(left_frame, weight=3)
        main_pane.add(right_frame, weight=1)

        # ----- Панель кнопок (ДВА РЯДА) -----
        btns_container = tk.Frame(left_frame)
        btns_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 6))

        # Ряд 1: действия
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

        # Ряд 2: опции/скорость
        row2 = tk.Frame(btns_container)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        self.viz_cb = tk.Checkbutton(row2, text="Визуализация", variable=self.visualize)
        self.viz_cb.pack(side=tk.LEFT, padx=5, pady=2)

        speed_frame = tk.Frame(row2)
        speed_frame.pack(side=tk.LEFT, padx=10, pady=2)
        tk.Label(speed_frame, text="Скорость:").pack(side=tk.LEFT)
        self.speed_scale = tk.Scale(
            speed_frame, from_=100, to=2000, orient=tk.HORIZONTAL,
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

        # ----- Правая часть: панель статусов + вертикальный сплиттер (логи/результаты) -----
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

    # ------------------------- Вспомогательные «мьюты» визуализации -------------------------

    def viz_draw(
        self,
        highlight_vertices: Optional[Set[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Вызывать вместо прямого draw_graph в пошаговых местах."""
        if self.visualize.get():
            self.draw_graph(highlight_vertices=highlight_vertices, highlight_edges=highlight_edges)

    def viz_update(self) -> None:
        """Вызывать вместо root.update* в пошаговых местах."""
        if self.visualize.get():
            self.root.update_idletasks()

    # ------------------------- Алгоритм Куна (BFS) -------------------------

    def prepare_algorithm(self) -> None:
        """Подготовка: сброс состояния и запуск BFS из свободных левых вершин."""
        if not self.graph:
            messagebox.showwarning("Предупреждение", "Сначала загрузите граф!")
            return

        self.add_log("=" * 50)
        self.add_log("ПОДГОТОВКА АЛГОРИТМА КУНА")
        self.add_log("=" * 50)

        # Сброс
        self.matching.clear()
        self.current_step = 0
        self.steps.clear()
        self.auto_running = False
        self.current_start = None

        # Стартовать только из СВОБОДНЫХ левых
        self.left_vertices = sorted(
            [u for u in self.left_partition if u not in self.matching.values()],
            key=int,
        )
        self.current_vertex_index = 0

        if self.current_vertex_index < len(self.left_vertices):
            self.current_start = self.left_vertices[self.current_vertex_index]
            self.initialize_bfs(self.current_start)

        self.step_btn.config(state=tk.NORMAL)
        self.auto_btn.config(state=tk.NORMAL)
        self.update_status("Алгоритм подготовлен. Готов к выполнению.")
        self.update_step_counter()

    def initialize_bfs(self, start_vertex: str) -> None:
        """Инициализация очереди, посещений и предков для очередной левой вершины."""
        self.bfs_queue = deque([start_vertex])
        self.visited = {start_vertex}
        self.parent = {start_vertex: None}
        self.path_found = False
        self.end_vertex: Optional[str] = None
        self.add_log(f"Инициализирован BFS для вершины {start_vertex}")

    def bfs_step(self) -> bool:
        """Один шаг BFS. Возвращает True, если найден свободный правый конец (увелич. путь)."""
        if not self.bfs_queue:
            return False

        u = self.bfs_queue.popleft()
        self.add_log(f"Обрабатываем вершину {u}")
        self.viz_draw(highlight_vertices={u})
        self.viz_update()

        for v in sorted(self.graph.get(u, []), key=int):
            if v in self.visited:
                continue

            self.visited.add(v)
            self.add_log(f"Проверяем ребро {u}-{v}")
            self.viz_draw(highlight_vertices={u, v}, highlight_edges=[(u, v)])
            self.viz_update()

            # Правый свободный — нашли путь
            if v in self.right_partition and v not in self.matching:
                self.path_found = True
                self.end_vertex = v
                self.parent[v] = u
                self.add_log(f"Вершина {v} свободна! Путь найден")
                return True

            # Правый занят — идём к его партнёру слева
            if v in self.right_partition and v in self.matching:
                w = self.matching[v]  # партнёр слева
                if w not in self.visited:
                    self.bfs_queue.append(w)
                    self.parent[w] = v
                    self.visited.add(w)
                    self.add_log(f"Вершина {v} соединена с {w}, продолжаем поиск")

        return False

    def find_augmenting_path(self) -> Optional[List[Tuple[str, str]]]:
        """Полный проход BFS до нахождения пути или опустошения очереди."""
        while self.bfs_queue:
            if self.bfs_step():
                return self.reconstruct_path()
        return None

    def reconstruct_path(self) -> List[Tuple[str, str]]:
        """Восстановление увеличивающего пути как списка рёбер (left, right)."""
        path: List[Tuple[str, str]] = []
        current = self.end_vertex
        while current is not None:
            parent = self.parent[current]
            if parent is not None:
                path.append((parent, current))
            current = parent
        path.reverse()
        return path

    def unmatch_left(self, u: str) -> None:
        """Если левая вершина u уже сматчена с какой-то правой — снять это соответствие."""
        to_delete: Optional[str] = None
        for r, l in self.matching.items():
            if l == u:
                to_delete = r
                break
        if to_delete is not None:
            del self.matching[to_delete]

    def next_step(self, from_auto: bool = False) -> None:
        """Следующий шаг. В автозапуске не сбрасывает цикл; по клику — останавливает авто."""
        if not from_auto and self.auto_running:
            self.auto_btn.config(text="Автозапуск")
            self.auto_running = False
            return

        if self.current_start is None:
            self.add_log("Алгоритм завершен")
            self.finalize_algorithm()
            return

        path = self.find_augmenting_path()

        if path:
            self.add_log(
                "Найден увеличивающий путь: " +
                " -> ".join([f"{u}-{v}" for u, v in path])
            )

            # Флиппинг с контролем уникальности слева
            for u, v in path:
                if v in self.matching and self.matching[v] == u:
                    del self.matching[v]
                else:
                    self.unmatch_left(u)
                    self.matching[v] = u

            self.add_log(f"Обновлено паросочетание. Новый размер: {len(self.matching)}")
            self.viz_draw()
            self._advance_to_next_left()
        else:
            self.add_log(f"Не удалось найти увеличивающий путь для вершины {self.current_start}")
            self._advance_to_next_left()

        self.update_step_counter()

        if self.current_start is None:
            self.finalize_algorithm()

    def _advance_to_next_left(self) -> None:
        """Переход к следующей стартовой левой вершине, пропуская уже сматченных."""
        self.current_vertex_index += 1
        while (
            self.current_vertex_index < len(self.left_vertices)
            and self.left_vertices[self.current_vertex_index] in self.matching.values()
        ):
            self.current_vertex_index += 1

        if self.current_vertex_index < len(self.left_vertices):
            self.current_start = self.left_vertices[self.current_vertex_index]
            self.initialize_bfs(self.current_start)
        else:
            self.current_start = None

    def finalize_algorithm(self) -> None:
        """Подсчёт и вывод результатов: размер, «полное из меньшей доли», совершенство."""
        self.add_log("=" * 50)
        self.add_log("АЛГОРИТМ ЗАВЕРШЁН")

        L = len(self.left_partition)
        R = len(self.right_partition)
        k = len(self.matching)

        complete_from_smaller = (k == min(L, R))
        perfect = (L == R) and (k == L)

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
        self.result_text.insert(tk.END, "Рёбра паросочетания:\n")
        for right, left in sorted(self.matching.items(), key=lambda x: (int(x[0]), int(x[1]))):
            self.result_text.insert(tk.END, f"{left} - {right}\n")

        self.step_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)
        self.update_status("Алгоритм завершен")

        # Если это был тест — показать ожидания
        if self._last_test_name and self._last_test_name in self.test_expectations:
            exp = self.test_expectations[self._last_test_name]
            if exp.get("bipartite", True):
                self.add_log(
                    f"[Ожидаемо] размер: {exp['size']}, "
                    f"полное: {'ДА' if exp['complete'] else 'НЕТ'}, "
                    f"совершенное: {'ДА' if exp['perfect'] else 'НЕТ'}"
                )

    def toggle_auto_run(self) -> None:
        """Тумблер автозапуска шагов с задержкой delay."""
        if not self.auto_running:
            if self.current_start is None:
                return
            self.auto_btn.config(text="Остановить")
            self.auto_running = True
            self._auto_run_tick()
        else:
            self.auto_btn.config(text="Автозапуск")
            self.auto_running = False

    def _auto_run_tick(self) -> None:
        """Один тик автозапуска."""
        if self.auto_running and self.current_start is not None:
            self.next_step(from_auto=True)
            self.root.after(self.delay.get(), self._auto_run_tick)
        elif self.current_start is None:
            self.auto_btn.config(text="Автозапуск")
            self.auto_running = False

    # ------------------------- Логи/статусы/визуализация -------------------------

    def add_log(self, message: str) -> None:
        """Добавить строку в логи."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        if self.visualize.get():
            self.root.update_idletasks()

    def update_status(self, message: str) -> None:
        """Обновить строку статуса."""
        self.status_var.set(message)
        if self.visualize.get():
            self.root.update_idletasks()

    def update_step_counter(self) -> None:
        """Обновить счётчик шага (текущая/всего стартовых слева)."""
        total = len(self.left_vertices) if self.left_vertices else 0
        current_idx = self.current_vertex_index
        current_start = self.current_start
        current = min(current_idx + (1 if current_start is not None else 0), total) if total else 0
        self.step_var.set(f"{current}/{total}")

    def clear_graph(self) -> None:
        """Полный сброс графа и полей вывода."""
        self.graph.clear()
        self.left_partition.clear()
        self.right_partition.clear()
        self.matching.clear()
        self.vertex_count = 0
        self.left_vertices = []
        self.current_vertex_index = 0
        self.current_start = None
        self.log_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.update_status("Граф очищен")
        self.step_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)
        self.find_btn.config(state=tk.DISABLED)
        self.viz_draw()

    def draw_graph(
        self,
        highlight_vertices: Optional[Set[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Отрисовка двудольного графа с подсветкой вершин/рёбер и текущего паросочетания."""
        self.ax.clear()

        if not self.graph:
            self.ax.text(0.5, 0.5, "Граф не загружен", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return

        highlight_vertices = set(highlight_vertices or [])
        highlight_edges = set(highlight_edges or [])

        # Координаты вершин: левая доля x=0, правая x=3
        pos: Dict[str, Tuple[int, int]] = {}
        for i, node in enumerate(sorted(self.left_partition, key=int)):
            pos[node] = (0, i)
        for i, node in enumerate(sorted(self.right_partition, key=int)):
            pos[node] = (3, i)

        # Все рёбра (серые)
        for u, nbrs in self.graph.items():
            for v in nbrs:
                if u in pos and v in pos and int(u) < int(v):  # чтобы не рисовать дважды
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    self.ax.plot([x1, x2], [y1, y2], "gray", alpha=0.5)

        # Рёбра матчинга (красные)
        for right, left in self.matching.items():
            if left in pos and right in pos:
                x1, y1 = pos[left]
                x2, y2 = pos[right]
                self.ax.plot([x1, x2], [y1, y2], "red", linewidth=3)

        # Подсвеченные рёбра (оранжевые, пунктир)
        for u, v in highlight_edges:
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                self.ax.plot([x1, x2], [y1, y2], "orange", linewidth=3, linestyle="dashed")

        # Вершины (обычные)
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

        # Подсвеченные вершины
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
        self.ax.set_title("Двудольный граф — алгоритм Куна")
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------- Загрузка / парсинг -------------------------

    def load_graph(self) -> None:
        """Диалог выбора файла и загрузка графа."""
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
        """Парсинг и применение матрицы из текста."""
        vertex_count, matrix = self._parse_graph_text(text)
        self._apply_matrix(matrix, vertex_count, source_name)

    def _parse_graph_text(self, text: str) -> Tuple[int, List[List[str]]]:
        """Разбор текстового представления графа (валидация формата)."""
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
        """Построение списка смежности, проверка двудольности и подготовка UI."""
        # Словарь смежности
        self.graph = {str(i): [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # игнорируем самопетли
                if matrix[i][j] != "-":  # любое не '-' — есть ребро
                    u, v = str(i), str(j)
                    self.graph[u].append(v)

        self.vertex_count = n

        # Проверка двудольности
        self.add_log(f"Проверка двудольности графа ({source_name})...")
        if not self.is_bipartite():
            self.add_log("ОШИБКА: Граф не является двудольным!")
            messagebox.showerror("Ошибка", "Граф не является двудольным!")
            # Откат состояния
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
        self.steps.clear()
        self.current_step = 0

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
        """Проверка двудольности (BFS-окраска), заполнение левой/правой долей."""
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

        # Изолированные вершины — в левую (на корректность не влияет)
        for i in range(self.vertex_count):
            si = str(i)
            if si not in self.left_partition and si not in self.right_partition:
                self.left_partition.add(si)

        return True

    # ------------------------- Тесты -------------------------

    def _make_tests_data(self) -> Dict[str, str]:
        """Возвращает словарь встроенных тестов: имя -> текст-граф."""
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
        """Ожидаемые результаты для тестов (для валидации корректности)."""
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
        """Загрузка выбранного теста и вывод ожидаемого результата в логах."""
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
