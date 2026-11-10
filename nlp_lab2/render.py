"""Текстовый вывод и автономный HTML-отчёт с деревьями SVG."""

import html
import textwrap

from lexer import Token


def text_tree(node, depth=0):
    """Иерархия через отступы, без скобок и псевдографики."""
    if isinstance(node, Token):
        return "  " * depth + "«" + node.text + "»"
    lines = ["  " * depth + node.label]
    lines.extend(text_tree(child, depth + 1) for child in node.children)
    return "\n".join(lines)


def text_result(record):
    heading = f"Строка {record['line']}. Запрос: {record['query']}"
    if record["tree"] is not None:
        return heading + "\nРезультат: успех\n\n" + text_tree(record["tree"])
    error = record["error"]
    column = error["column"]
    pointer = " " * len(record["query"][:column - 1].expandtabs(4)) + "^"
    return (
        heading + "\nРезультат: неудача\n"
        + f"{error['stage']} ошибка. Токен №{error['token_number']}, символ №{column}: "
        + f"«{error['actual']}».\n"
        + error["message"] + "\nОжидалось: " + ", ".join(error["expected"])
        + "\n" + record["query"].expandtabs(4) + "\n" + pointer
    )


def svg_tree(root):
    """Размещение по ширине поддеревьев: соседние ветви не накладываются."""
    gap = 20
    levels = {}

    def measure(node, depth):
        leaf = isinstance(node, Token)
        label = node.text if leaf else node.label
        lines = textwrap.wrap(label, width=18, replace_whitespace=True) or [""]
        width = max(74, max(map(len, lines)) * 9 + 24)
        height = 20 * len(lines) + 20
        levels[depth] = max(levels.get(depth, 0), height)
        children = [] if leaf else [measure(c, depth + 1) for c in node.children]
        subtree_width = max(width, sum(c["span"] for c in children) + gap * max(0, len(children) - 1))
        return {"lines": lines, "width": width, "height": height,
                "span": subtree_width, "children": children, "leaf": leaf, "depth": depth}

    layout = measure(root, 0)
    level_y = {}
    y = 24
    for depth in sorted(levels):
        level_y[depth] = y
        y += levels[depth] + 56

    edges, boxes = [], []

    def draw(item, left):
        center = left + item["span"] / 2
        top = level_y[item["depth"]]
        child_span = sum(c["span"] for c in item["children"]) + gap * max(0, len(item["children"]) - 1)
        child_left = center - child_span / 2
        for child in item["children"]:
            child_center = child_left + child["span"] / 2
            bottom = top + item["height"]
            child_top = level_y[child["depth"]]
            middle = (bottom + child_top) / 2
            edges.append(
                f'<path d="M {center} {bottom} C {center} {middle}, '
                f'{child_center} {middle}, {child_center} {child_top}" '
                'fill="none" stroke="#94a3b8" stroke-width="1.5"/>'
            )
            draw(child, child_left)
            child_left += child["span"] + gap
        fill = "#e8f1ff" if item["leaf"] else "#ffffff"
        stroke = "#9abbee" if item["leaf"] else "#cbd5e1"
        boxes.append(
            f'<rect x="{center - item["width"] / 2}" y="{top}" '
            f'width="{item["width"]}" height="{item["height"]}" rx="8" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )
        for line_index, line in enumerate(item["lines"]):
            boxes.append(
                f'<text x="{center}" y="{top + 25 + line_index * 20}" '
                'text-anchor="middle" font-family="monospace" font-size="13" '
                f'fill="#17253b">{html.escape(line)}</text>'
            )

    draw(layout, 24)
    width = layout["span"] + 48
    height = y - 56 + 24
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Дерево разбора">'
        '<title>Дерево разбора запроса. Слова находятся в голубых блоках.</title>'
        + "".join(edges) + "".join(boxes) + "</svg>"
    )


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#17253b;
font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:36px auto;padding:0 24px 60px}
h1{font-size:30px;line-height:1.2;margin:0 0 12px}h2{font-size:18px;margin:0}
.muted{color:#52647c}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}
.stat{background:white;border:1px solid #dce3ed;border-radius:12px;padding:12px 20px}
.stat b{font-size:23px;margin-right:8px}.actions{display:flex;gap:10px;margin:16px 0}
button{border:1px solid #cbd5e1;border-radius:8px;padding:8px 12px;background:white;
color:#17253b;cursor:pointer;font:inherit}button:hover{background:#e8f1ff}
article{background:#fff;border:1px solid #dce3ed;border-radius:14px;margin:18px 0;padding:22px}
.badge{display:inline-block;border-radius:20px;font-size:13px;padding:3px 10px;margin-bottom:10px}
.ok{background:#e1f4e9;color:#16623a}.bad{background:#ffe8e6;color:#9f2722}
.query{font-size:18px;white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0 12px}
details{border-top:1px solid #e7ecf3;padding-top:12px}summary{cursor:pointer;font-weight:600}
.diagram{overflow:auto;margin-top:12px;border-radius:8px;background:#f8fafd;border:1px solid #e7ecf3}
.diagram svg{display:block;max-width:100%;height:auto;margin:0 auto}
.diagram.natural svg{max-width:none}.error{color:#9f2722}
pre{overflow:auto;background:#f8fafd;padding:12px;border-radius:8px;font-size:14px}
.legend{font-size:14px;color:#52647c}mark{background:#ffe0dc;color:#9f2722}
@media(max-width:600px){main{padding:0 12px}article{padding:16px}h1{font-size:25px}}
@media print{button{display:none}main{margin:0}.diagram{overflow:visible}article{break-inside:avoid}}
"""


def html_report(records):
    ok = sum(r["tree"] is not None for r in records)
    cards = []
    for index, record in enumerate(records):
        query = html.escape(record["query"])
        if record["tree"] is not None:
            badge = '<span class="badge ok">Успех</span>'
            opened = " open" if index < 2 else ""
            body = (
                f'<details{opened}><summary>Дерево разбора</summary>'
                '<p class="legend">Белые блоки — конструкции грамматики, голубые — '
                'исходные токены.</p>'
                '<p><button onclick="zoomTree(this)">Увеличить дерево</button></p>'
                '<div class="diagram">' + svg_tree(record["tree"]) + '</div></details>'
            )
        else:
            badge = '<span class="badge bad">Неудача</span>'
            e = record["error"]
            start = e["column"] - 1
            end = min(len(record["query"]), start + max(1, len(e["actual"])))
            highlighted = (
                html.escape(record["query"][:start]) + '<mark>'
                + html.escape(record["query"][start:end] or "[конец запроса]") + '</mark>'
                + html.escape(record["query"][end:])
            )
            body = (
                f'<p class="error">{html.escape(e["stage"])} ошибка: токен №{e["token_number"]}, '
                f'символ №{e["column"]}.</p><p>{html.escape(e["message"])}</p>'
                '<p><b>Ожидалось:</b> ' + html.escape(", ".join(e["expected"])) + '</p>'
                '<pre>' + highlighted + '</pre>'
            )
        cards.append(
            f'<article>{badge}<h2>Строка {record["line"]}</h2>'
            f'<p class="query">{query}</p>{body}</article>'
        )
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Парфюмерия — синтаксический анализ</title><style>' + STYLE + '</style></head>'
        '<body><main><p class="muted">Обработка естественного языка · Лабораторная №2</p>'
        '<h1>Запросы к каталогу парфюмерии</h1>'
        '<p class="muted">Разбор по формальной грамматике методом рекурсивного спуска.</p>'
        f'<div class="stats"><div class="stat"><b>{len(records)}</b>запросов</div>'
        f'<div class="stat"><b>{ok}</b>корректных</div>'
        f'<div class="stat"><b>{len(records) - ok}</b>с ошибками</div></div>'
        '<div class="actions"><button onclick="setTrees(true)">Раскрыть деревья</button>'
        '<button onclick="setTrees(false)">Свернуть деревья</button></div>'
        + "".join(cards)
        + '</main><script>function setTrees(value){document.querySelectorAll("details")'
        '.forEach(function(item){item.open=value;});}'
        'function zoomTree(button){const box=button.closest("details").querySelector(".diagram");'
        'const large=box.classList.toggle("natural");'
        'button.textContent=large?"Вписать по ширине":"Увеличить дерево";'
        'box.scrollLeft=(box.scrollWidth-box.clientWidth)/2;}'
        '</script></body></html>'
    )
