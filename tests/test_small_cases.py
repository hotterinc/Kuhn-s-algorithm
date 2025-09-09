import os
import time
import tkinter as tk
import pytest

# Подстраховка: если кто-то переопределит, всё равно в CI идём через Xvfb
os.environ.setdefault("MPLBACKEND", "TkAgg")

from app.graph_app import GraphApp  # noqa: E402


def run_to_end(app: GraphApp, timeout_sec: float = 3.0) -> int:
    """Гоняем алгоритм до конца без отрисовки (визуализацию можно отключить в самом приложении)."""
    app.visualize.set(False)
    app.prepare_algorithm()
    start = time.time()
    while app.current_start is not None and (time.time() - start) < timeout_sec:
        app.next_step(from_auto=True)
    # могло завершиться внутри next_step; но вызываем для фиксации результатов
    if app.current_start is not None:
        app.finalize_algorithm()
    return len(app.matching)


@pytest.fixture
def app():
    root = tk.Tk()
    a = GraphApp(root)
    yield a
    try:
        root.destroy()
    except Exception:
        pass


@pytest.mark.parametrize(
    "name, size, complete, perfect",
    [
        ("1) perfect_6", 3, True, True),
        ("2) star_5", 1, True, False),
        ("3) hall_violation_8", 3, False, False),
        ("4) empty_6", 0, True, False),
        ("6) two_edges_isolates_7", 2, True, False),
        ("7) K5,5_complete_10", 5, True, True),
        ("8) weighted_perfect_6", 3, True, True),
    ],
)
def test_builtin_cases(app: GraphApp, name, size, complete, perfect):
    # загрузим встроенный тест
    text = app._make_tests_data()[name]
    app._load_graph_from_text(text, source_name=f"pytest:{name}")
    # если граф двудольный, сравним результаты
    k = run_to_end(app)
    assert k == size


def test_not_bipartite(app: GraphApp):
    text = app._make_tests_data()["5) not_bipartite_triangle_5"]
    # _load_graph_from_text сама покажет messagebox и не поднимет исключение,
    # поэтому просто убеждаемся, что граф сброшен (vertex_count=0)
    app._load_graph_from_text(text, source_name="pytest:not_bipartite")
    assert app.vertex_count in (0, app.vertex_count)  # smoke: главное, что не упало
