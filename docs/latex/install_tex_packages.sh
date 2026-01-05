#!/bin/bash
# Script to install required LaTeX packages for main.tex compilation

echo "Installing required LaTeX packages..."
echo "This script requires sudo privileges."

# Install fontspec (part of texlive-fontsextra)
sudo pacman -S --noconfirm texlive-fontsextra

# Install biber for bibliography processing
sudo pacman -S --noconfirm biber

# Install biblatex support (part of texlive-bibtexextra)
sudo pacman -S --noconfirm texlive-bibtexextra

# Install additional packages that might be needed
sudo pacman -S --noconfirm texlive-latexextra

echo ""
echo "Verifying installations..."
echo "Checking fontspec:"
kpsewhich fontspec.sty

echo "Checking biber:"
which biber

echo "Checking biblatex:"
kpsewhich biblatex.sty

echo ""
echo "Installation complete! You can now compile main.tex using:"
echo "  latexmk -pdf -xelatex main.tex"



