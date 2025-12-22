# Thesis LaTeX Template

This is a template/skeleton version of the thesis LaTeX structure.

## Structure

- `main.tex` - Main document file
- `abstract.tex` - Abstract chapter
- `introduction.tex` - Introduction chapter
- `materials_methods.tex` - Materials and Methods chapter
- `result.tex` - Results chapter
- `discussion.tex` - Discussion chapter
- `conclusion.tex` - Conclusion chapter
- `appendix.tex` - Title page, acknowledgments, table of contents, etc.
- `references.bib` - Bibliography file

## Directories

- `figures/` - Place your compiled figures/images (PDF, PNG, etc.) here
- `figures-tex/` - Source files for TikZ diagrams (see example workflow diagram)
- `build/` - Build output directory (auxiliary files)

## Requirements

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **LaTeX Distribution**: MiKTeX (Windows), MacTeX (macOS), or TeX Live (Linux/Windows/macOS)
- **PowerShell**: Version 5.1 or later (for Windows build script)
- **Perl**: Required for `latexmk` (usually included with LaTeX distributions)
- **Disk Space**: At least 2-3 GB for a basic LaTeX installation

### Required LaTeX Packages

The template uses the following LaTeX packages (will be automatically installed by your LaTeX distribution if missing):
- `fontspec` - Font selection for XeLaTeX
- `geometry` - Page layout
- `graphicx` - Graphics inclusion
- `setspace` - Line spacing
- `ragged2e` - Text alignment
- `fancyhdr` - Headers and footers
- `amsmath`, `amssymb` - Mathematical symbols
- `booktabs` - Professional tables
- `biblatex` - Bibliography management
- `titlesec` - Title formatting
- `caption` - Caption customization
- `tikz` - For creating diagrams (if using TikZ)

## Installation

### Windows: Installing MiKTeX

1. **Download MiKTeX**:
   - Visit [https://miktex.org/download](https://miktex.org/download)
   - Download the MiKTeX Installer (Basic or Complete)
   - Basic installer (~150 MB) downloads packages on-demand
   - Complete installer (~4 GB) includes all packages

2. **Install MiKTeX**:
   - Run the installer executable
   - Choose installation directory (default is recommended)
   - Select "Install missing packages on-the-fly: Yes" (recommended)
   - Complete the installation

3. **Verify Installation**:
   ```powershell
   xelatex --version
   latexmk --version
   biber --version
   ```

4. **Update MiKTeX** (if needed):
   ```powershell
   # Open MiKTeX Console (Start Menu > MiKTeX > Maintenance)
   # Or use command line:
   mpm --update-db
   mpm --update
   ```

### macOS: Installing MacTeX

1. **Download MacTeX**:
   - Visit [https://www.tug.org/mactex/](https://www.tug.org/mactex/)
   - Download MacTeX.pkg (full distribution, ~4 GB)
   - Or MacTeX-Basic.pkg (smaller, ~100 MB) for minimal installation

2. **Install MacTeX**:
   - Open the downloaded `.pkg` file
   - Follow the installation wizard
   - Installation may take 30-60 minutes

3. **Verify Installation**:
   ```bash
   xelatex --version
   latexmk --version
   biber --version
   ```

### Linux: Installing TeX Live

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install texlive-full
# Or for minimal installation:
sudo apt-get install texlive-xetex texlive-bibtex-extra biber
```

#### Fedora/RHEL:
```bash
sudo dnf install texlive-scheme-full
# Or for minimal installation:
sudo dnf install texlive-xetex texlive-biblatex biber
```

#### Arch Linux:
```bash
sudo pacman -S texlive-most texlive-lang
```

#### Verify Installation:
```bash
xelatex --version
latexmk --version
biber --version
```

### Installing Additional Tools

#### Perl (for latexmk)
- **Windows**: Usually included with MiKTeX, or download from [Strawberry Perl](https://strawberryperl.com/)
- **macOS**: Usually pre-installed, or install via Homebrew: `brew install perl`
- **Linux**: Usually pre-installed, or install via package manager: `sudo apt-get install perl`

#### PowerShell (for build script on Windows)
- **Windows**: Pre-installed on Windows 10/11
- **macOS/Linux**: Install PowerShell Core: [https://github.com/PowerShell/PowerShell](https://github.com/PowerShell/PowerShell)

### Installing Missing Packages

If you encounter missing package errors during compilation:

**MiKTeX (Windows)**:
- Packages are automatically installed on first use
- Or use MiKTeX Console: Start Menu > MiKTeX > Maintenance > Package Manager

**MacTeX/TeX Live**:
- Use package manager:
  ```bash
  tlmgr install <package-name>
  ```
- Update package database:
  ```bash
  tlmgr update --self
  tlmgr update --all
  ```

## Quick Start

1. **Install LaTeX distribution** (see Installation section above)
2. **Verify installation**:
   ```powershell
   # Windows PowerShell
   xelatex --version
   
   # macOS/Linux Terminal
   xelatex --version
   ```
3. **Compile the template**:
   ```powershell
   .\build.ps1 -TexFile main.tex
   ```
4. **Open the generated PDF**: `main.pdf`

## Building

To compile the document, use the build script:

```powershell
.\build.ps1 -TexFile main.tex
```

Or use latexmk directly:

```bash
latexmk -pdf -outdir=build main.tex
```

## Build Script (`build.ps1`) Documentation

The `build.ps1` script automates LaTeX compilation with intelligent engine detection and file management.

### Script Parameters

- **`-TexFile`** (Required): Path to the `.tex` file to compile
- **`-Engine`** (Optional, default: "auto"): LaTeX engine to use
  - `"auto"` - Automatically detects based on packages used
  - `"xelatex"` - XeLaTeX (for Unicode fonts like Times New Roman)
  - `"lualatex"` - LuaLaTeX
  - `"pdflatex"` - PDFLaTeX (default)
- **`-Watch`** (Optional): Enable watch mode for auto-recompilation on file changes
- **`-OutputToFigures`** (Optional): Output PDF to `figures/` directory (useful for diagram compilation)

### Functions

#### `Find-LatexRoot`
Finds the root LaTeX directory by walking up the directory tree until it finds a `figures/` directory. Used when `-OutputToFigures` is enabled to determine where to place the compiled PDF.

#### `Move-Pdf`
Moves the compiled PDF from the `build/` directory to the main directory. Ensures the final PDF is in the correct location while keeping auxiliary files organized.

### Engine Auto-Detection

The script automatically detects which LaTeX engine to use by scanning the `.tex` file:
- If `\usepackage{fontspec}` is found → uses **XeLaTeX**
- If `\usepackage{luatextra}` is found → uses **LuaLaTeX**
- Otherwise → uses **PDFLaTeX**

### Build Modes

#### Normal Build Mode
Standard one-time compilation:
- Compiles the document using `latexmk`
- Places auxiliary files (`.aux`, `.log`, `.toc`, etc.) in `build/` directory
- Moves final PDF to main directory
- Provides success/failure feedback

#### Watch Mode (`-Watch`)
Continuous compilation mode:
- Sets up a file system watcher
- Runs `latexmk -pvc` (preview continuously)
- Automatically recompiles when source files change
- Moves updated PDF automatically
- Press `Ctrl+C` to stop

#### OutputToFigures Mode (`-OutputToFigures`)
For compiling standalone diagram files:
- Compiles the `.tex` file
- Moves PDF to `figures/` directory
- Cleans up all auxiliary files
- Removes empty build directory
- Keeps only the final PDF

### Usage Examples

```powershell
# Normal compilation
.\build.ps1 -TexFile main.tex

# Compile with watch mode (auto-recompile on save)
.\build.ps1 -TexFile main.tex -Watch

# Compile a diagram and output to figures/
.\build.ps1 -TexFile figures-tex/diagram.tex -OutputToFigures

# Force specific engine
.\build.ps1 -TexFile main.tex -Engine xelatex

# Combine options
.\build.ps1 -TexFile main.tex -Watch -Engine xelatex
```

### File Organization

- **Auxiliary files**: All `.aux`, `.log`, `.toc`, `.bbl`, `.bcf`, `.fls`, `.fdb_latexmk`, `.xdv` files go to `build/` directory
- **Final PDF**: Moved to main directory (or `figures/` if `-OutputToFigures`)
- **Clean workspace**: Main directory stays clean, only containing source files and final PDF

### Error Handling

- Checks compilation success via exit codes
- Shows error messages with log file location
- Displays last 10 lines of log file on failure for debugging

## Creating TikZ Diagrams

The template includes an example TikZ workflow diagram. To create your own:

1. **Create a diagram file** in `figures-tex/` directory (see `example_workflow.tex` for reference)
2. **Compile it** using:
   ```powershell
   .\build.ps1 -TexFile figures-tex/your_diagram.tex -OutputToFigures
   ```
3. **Include it** in your document:
   ```latex
   \begin{figure}[htbp]
       \centering
       \includegraphics[width=0.8\textwidth]{figures/your_diagram.pdf}
       \caption{Your caption here.}
       \label{fig:your_diagram}
   \end{figure}
   ```

See `figures-tex/README.md` for more details and TikZ resources.

## Notes

- This template uses XeLaTeX (for Times New Roman font support)
- All content sections are marked with comments indicating where to add your content
- Replace placeholder text (YOUR NAME, YOUR UNIVERSITY, etc.) with your actual information
- Add your figures to the `figures/` directory
- Add bibliography entries to `references.bib`
- See `materials_methods.tex` for an example of including a TikZ diagram
