# TikZ Diagram Source Files

This directory contains the source `.tex` files for TikZ diagrams.

## How to Create and Compile Diagrams

1. **Create a new TikZ diagram file** in this directory (e.g., `my_diagram.tex`)
   - Use the `standalone` document class
   - Include necessary TikZ libraries
   - Define your diagram

2. **Compile the diagram** using the build script:
   ```powershell
   .\build.ps1 -TexFile figures-tex/my_diagram.tex -OutputToFigures
   ```
   This will:
   - Compile the diagram
   - Generate a PDF in the `figures/` directory
   - Clean up auxiliary files

3. **Include the diagram** in your LaTeX document:
   ```latex
   \begin{figure}[htbp]
       \centering
       \includegraphics[width=0.8\textwidth]{figures/my_diagram.pdf}
       \caption{Your caption here.}
       \label{fig:my_diagram}
   \end{figure}
   ```

## Example File

See `example_workflow.tex` for a complete example of a TikZ flowchart diagram.

## TikZ Resources

- [TikZ & PGF Manual](https://tikz.dev/)
- [TikZ Examples Gallery](https://texample.net/tikz/)
- [Overleaf TikZ Tutorial](https://www.overleaf.com/learn/latex/TikZ_package)

