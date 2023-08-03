import os
import pandas as pd

cwd = os.getcwd()

def image(experiment, slice, variable, fold, model):
    imdir = "../img"
    path = os.path.join(imdir, experiment, slice, variable, fold, model)
    imgpath = "../../img" 
    text = ""
    for image in os.listdir(path):
        print(image)
        text += '''\\begin{figure}[H]\n\\centering'''
        text += '''\t\\includegraphics[width=0.5\\textwidth]{'''
        text += os.path.join(imgpath, experiment, slice, variable, fold, model, image)
        text += "}\n"
        text += '''\\end{figure}\n'''
    return text



def table(experiment, slice, variable, fold, col0, col1):
    path = os.path.join(cwd, experiment, slice, variable, fold)
    df = pd.read_csv(path)

    columns = [df.columns[1]]
    for c in df.columns[2+col0:2+col1]:
        columns.append(c)

    tex = '''\\begin{center}\n\\begin{tabular}{c''' 
    tex += "|c" * (len(columns) - 1) + "}\n"

    for colidx, col in enumerate(columns):
        tex += "\\text{" + col + "}"
        if colidx != len(columns) - 1:
            tex += " & "
    tex += "\\\\ \hline \n"

    for indx, row in df.iterrows():
        for colidx, col in enumerate(columns):
            value = row[col]

            if isinstance(value, float):
                value = "%.4f"%value
                tex += "$" + value + "$"
            else:
                if value == "hybrid_mult":
                    value = "hybrid mult"
                tex += "\\text{" + value + "}"
            if colidx == len(columns) - 1:
                if indx != df.shape[0] - 1:
                    tex += "\\\\\n"
            else:
                tex += " & "

    tex = tex + '''\n\\end{tabular}\n\\end{center}\n'''
    return tex
    
tex = ""
with open("tables.tex", "w") as f:
    EXPRS = os.listdir(cwd)

    EXPRS = ["2A", "2B", "2C"]
    for expr in EXPRS:
        expr_dir = os.path.join(cwd, expr)
        if not os.path.isdir(expr_dir) or expr == "results":
            continue
        tex += "\section{Experiment " + expr + "}\n"
        for slice in os.listdir(expr_dir):
            slice_dir = os.path.join(expr_dir, slice)

            for var in os.listdir(slice_dir):
                tex += "\subsection{slice: " + slice + ", variables: " + var + "}\n"
                var_dir = os.path.join(slice_dir, var)

                for fold in os.listdir(var_dir):
                    tex += "\\textbf{Fold:} " + fold + "\n"
                    tex += table(expr, slice, var, fold, 0, 2)
                    tex += table(expr, slice, var, fold, 2, 4)
                    tex += table(expr, slice, var, fold, 4, 6)
                    tex += table(expr, slice, var, fold, 6, 8)
                    tex += table(expr, slice, var, fold, 8, 11)

                    f.write(tex)
                    tex = ""

tex = ""
with open("images.tex", "w") as f:
    EXPRS = os.listdir(cwd)

    EXPRS = ["2A", "2B", "2C"]
    for expr in EXPRS:
        expr_dir = os.path.join(cwd, expr)
        if not os.path.isdir(expr_dir) or expr == "results":
            continue
        tex += "\section{Experiment " + expr + "}\n"
        for slice in os.listdir(expr_dir):
            slice_dir = os.path.join(expr_dir, slice)

            for var in os.listdir(slice_dir):
                tex += "\subsection{slice: " + slice + ", variables: " + var + "}\n"
                var_dir = os.path.join(slice_dir, var)

                for fold in os.listdir(var_dir):
                    fold_no_extension, _ = os.path.splitext(fold)
                    path = os.path.join(cwd, expr, slice, var, fold)
                    df = pd.read_csv(path)

                    tex += "\\textbf{Fold:} " + fold + "\n"

                    for model in df["models"]:
                        tex += image(expr, slice, var, fold_no_extension, model)

                    f.write(tex)
                    tex = ""

with open("sumarized-tables.tex", "w") as f:
    EXPRS = os.listdir(cwd)

    EXPRS = ["2A", "2B", "2C"]
    for expr in EXPRS:
        expr_dir = os.path.join(cwd, expr)
        if not os.path.isdir(expr_dir) or expr == "results":
            continue
        tex += "\section{Experiment " + expr + "}\n"
        for slice in os.listdir(expr_dir):
            slice_dir = os.path.join(expr_dir, slice)

            for var in os.listdir(slice_dir):
                tex += "\subsection{slice: " + slice + ", variables: " + var + "}\n"
                var_dir = os.path.join(slice_dir, var)

                for fold in os.listdir(var_dir):
                    tex += "\\textbf{Fold:} " + fold + "\n"
                    tex += table(expr, slice, var, fold, 8, 11)
                    tex += table(expr, slice, var, fold, 4, 6)

                    f.write(tex)
                    tex = ""
