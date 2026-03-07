# LaTeX Project Structure

```
latex-project/
│
├── designs/                        # Reusable design/style files
│   └── academic-article/
│       └── design.sty              # Packages, margins, fonts — the "look"
│
├── papers/                         # One subfolder per paper
│   ├── mlir-hpc/
│   │   ├── main.tex                # Entry point — loads design + content
│   │   ├── content.tex             # The actual writing (sections, text)
│   │   ├── refs.bib                # (optional) bibliography
│   │   └── figures/                # Figures used only by this paper
│   │       ├── commands.png
│   │       └── compiler.png
│   │
│   └── paper2-example/
│       ├── main.tex
│       ├── content.tex
│       └── figures/
│
└── shared/
    └── figures/                    # Figures shared across multiple papers
```

## How to add a new paper

1. Copy any paper folder (e.g. `papers/paper2-example/`) and rename it.
2. Update the `\title`, `\author`, and `\date` in `main.tex`.
3. Write your content in `content.tex`.
4. Put figures in the local `figures/` folder.
5. Compile by running `pdflatex` from inside your paper folder:
   ```
   cd papers/my-new-paper
   pdflatex main.tex
   ```

## How to add a new design

1. Create a new folder under `designs/` (e.g. `designs/ieee-style/`).
2. Write your packages and settings into `design.sty`.
3. In any paper's `main.tex`, change the `\usepackage` line to point to the new design:
   ```latex
   \usepackage{../../designs/ieee-style/design}
   ```

## Rules of thumb

| What changes per paper?    | Where does it go?   |
|----------------------------|---------------------|
| Title, author, date        | `main.tex`          |
| Sections, text, equations  | `content.tex`       |
| Figures for this paper     | `figures/`          |
| Bibliography               | `refs.bib`          |

| What is shared?            | Where does it go?         |
|----------------------------|---------------------------|
| Margins, fonts, packages   | `designs/<name>/design.sty` |
| Figures used by many papers| `shared/figures/`         |
