# LaTeX Package Installation Guide

## Required Packages for main.tex

To compile `main.tex`, you need to install the following packages:

### Installation Command (run with sudo):

```bash
sudo pacman -S --noconfirm texlive-fontsextra biber texlive-bibtexextra texlive-latexextra
```

### What each package provides:

- **texlive-fontsextra**: Provides `fontspec` package (required for XeLaTeX font handling)
- **biber**: Bibliography processor (required for biblatex backend)
- **texlive-bibtexextra**: Provides `biblatex` package and related bibliography styles
- **texlive-latexextra**: Additional LaTeX packages (provides titlesec, caption, booktabs, etc.)

### Alternative: Install using the provided script

```bash
./install_tex_packages.sh
```

### Verify Installation

After installation, verify packages are available:

```bash
kpsewhich fontspec.sty
kpsewhich biblatex.sty
which biber
```

### Compile the Document

Once packages are installed, compile using:

```bash
latexmk -pdf -xelatex main.tex
```

Or for a clean compilation:

```bash
latexmk -c main.tex  # Clean auxiliary files
latexmk -pdf -xelatex main.tex  # Compile
```