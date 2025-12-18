if run test file that use TimeNewRoman, run: ` xelatex -interaction=nonstopmode test.tex `

if run main file, run: ` latexmk -pdf test.tex  `

if making a tex file to create a diagram, put the generated diagram into `` figures/ ``

for figures file, run 

`.\build.ps1 -TexFile figures-tex/direction_an_diagram.tex -OutputToFigures`

to set it build and put the correcspond diagram.pdf into figures