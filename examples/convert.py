import json

def convert_ipynb_to_py(ipynb_path, py_path=None):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    lines = []

    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            # Markdown → # 注释
            for line in cell["source"]:
                lines.append("# " + line.rstrip())
            lines.append("")  # 空行

        elif cell["cell_type"] == "code":
            # 保留代码
            for line in cell["source"]:
                lines.append(line.rstrip())
            lines.append("")  # 空行

    if py_path is None:
        py_path = ipynb_path.replace(".ipynb", ".py")

    with open(py_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved to {py_path}")
    return py_path


# 使用示例
convert_ipynb_to_py("LEdits.ipynb")

