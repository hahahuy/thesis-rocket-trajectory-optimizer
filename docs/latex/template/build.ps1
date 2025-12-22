# Build script: compiles LaTeX with aux files in build/, PDF in same dir as .tex
param(
    [Parameter(Mandatory=$true)]
    [string]$TexFile,
    
    [string]$Engine = "auto",
    
    [switch]$Watch = $false,
    
    [switch]$OutputToFigures = $false
)

# Get the base name and directory
$TexBaseName = [System.IO.Path]::GetFileNameWithoutExtension($TexFile)
$TexFullPath = Resolve-Path $TexFile
$TexDir = [System.IO.Path]::GetDirectoryName($TexFullPath)

# Handle case where file is in current directory
if ([string]::IsNullOrEmpty($TexDir)) {
    $TexDir = Get-Location
}

# Function to find root LaTeX directory (where figures/ should be)
function Find-LatexRoot {
    param([string]$StartDir)
    $current = $StartDir
    while ($current) {
        $figuresDir = Join-Path $current "figures"
        if (Test-Path $figuresDir) {
            return $current
        }
        $parent = [System.IO.Path]::GetDirectoryName($current)
        if ($parent -eq $current) {
            break
        }
        $current = $parent
    }
    # If not found, return the directory containing the .tex file
    return $StartDir
}

# If OutputToFigures is enabled, handle differently
if ($OutputToFigures) {
    $LatexRoot = Find-LatexRoot -StartDir $TexDir
    $FiguresDir = Join-Path $LatexRoot "figures"
    
    # Create figures directory if it doesn't exist
    if (-not (Test-Path $FiguresDir)) {
        New-Item -ItemType Directory -Path $FiguresDir | Out-Null
        Write-Host "Created figures directory: $FiguresDir" -ForegroundColor Yellow
    }
    
    # Use a temporary build directory in the tex file's directory
    $BuildDir = Join-Path $TexDir "build"
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir | Out-Null
    }
    
    Write-Host "Compiling diagram: $TexFile" -ForegroundColor Cyan
    Write-Host "Output PDF will go to: $FiguresDir" -ForegroundColor Yellow
    
    # Determine engine
    if ($Engine -eq "auto") {
        $content = Get-Content $TexFile -Raw
        if ($content -match "\\usepackage\{fontspec\}" -or $content -match "\\usepackage.*fontspec") {
            $Engine = "xelatex"
        } elseif ($content -match "\\usepackage\{luatextra\}" -or $content -match "\\usepackage.*luatextra") {
            $Engine = "lualatex"
        } else {
            $Engine = "pdflatex"
        }
    }
    
    # Compile
    Push-Location $TexDir
    try {
        $RelativeTexFile = [System.IO.Path]::GetFileName($TexFile)
        $CompileSuccess = $false
        
        if ($Engine -eq "xelatex") {
            latexmk -pdf -outdir=build -xelatex $RelativeTexFile
            if ($LASTEXITCODE -eq 0) { $CompileSuccess = $true }
        } elseif ($Engine -eq "lualatex") {
            latexmk -pdf -outdir=build -lualatex $RelativeTexFile
            if ($LASTEXITCODE -eq 0) { $CompileSuccess = $true }
        } else {
            latexmk -pdf -outdir=build $RelativeTexFile
            if ($LASTEXITCODE -eq 0) { $CompileSuccess = $true }
        }
        
        # Move PDF to figures directory
        $PdfInBuild = Join-Path $BuildDir "$TexBaseName.pdf"
        $PdfInFigures = Join-Path $FiguresDir "$TexBaseName.pdf"
        
        if (Test-Path $PdfInBuild) {
            Move-Item -Path $PdfInBuild -Destination $PdfInFigures -Force
            Write-Host "[SUCCESS] PDF created: $PdfInFigures" -ForegroundColor Green
            
            # Clean up all auxiliary files
            Write-Host "Cleaning up auxiliary files..." -ForegroundColor Yellow
            Get-ChildItem -Path $BuildDir -Filter "$TexBaseName.*" | Where-Object {
                $_.Extension -ne ".pdf"
            } | Remove-Item -Force -ErrorAction SilentlyContinue
            
            # Also clean up from tex file directory if any aux files were created there
            Get-ChildItem -Path $TexDir -Filter "$TexBaseName.*" | Where-Object {
                $_.Extension -match "\.(aux|log|out|toc|synctex|fdb_latexmk|fls|xdv)$"
            } | Remove-Item -Force -ErrorAction SilentlyContinue
            
            # Clean up empty build directory if it exists
            if ((Get-ChildItem -Path $BuildDir -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
                Remove-Item -Path $BuildDir -Force -ErrorAction SilentlyContinue
            }
            
            Write-Host "[SUCCESS] Cleanup complete. Only PDF remains in figures/ directory." -ForegroundColor Green
        } else {
            $LogFile = Join-Path $BuildDir "$TexBaseName.log"
            Write-Host "[ERROR] Compilation failed. Check $LogFile for errors." -ForegroundColor Red
            if (Test-Path $LogFile) {
                Write-Host "`nLast few lines of log file:" -ForegroundColor Yellow
                Get-Content $LogFile -Tail 10
            }
            exit 1
        }
    } finally {
        Pop-Location
    }
    
    exit 0
}

# Normal build mode - create build directory if it doesn't exist
$BuildDir = Join-Path $TexDir "build"
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Determine engine based on file content if auto
if ($Engine -eq "auto") {
    $content = Get-Content $TexFile -Raw
    if ($content -match "\\usepackage\{fontspec\}" -or $content -match "\\usepackage.*fontspec") {
        $Engine = "xelatex"
    } elseif ($content -match "\\usepackage\{luatextra\}" -or $content -match "\\usepackage.*luatextra") {
        $Engine = "lualatex"
    } else {
        $Engine = "pdflatex"
    }
}

# Function to move PDF from build to main directory
function Move-Pdf {
    param([string]$BuildDir, [string]$TexDir, [string]$TexBaseName)
    $PdfInBuild = Join-Path $BuildDir "$TexBaseName.pdf"
    if (Test-Path $PdfInBuild) {
        $PdfInMain = Join-Path $TexDir "$TexBaseName.pdf"
        Move-Item -Path $PdfInBuild -Destination $PdfInMain -Force
        Write-Host "[Updated] PDF moved to: $PdfInMain" -ForegroundColor Green
    }
}

Write-Host "Compiling $TexFile with engine: $Engine" -ForegroundColor Cyan
if ($Watch) {
    Write-Host "Watch mode: Auto-recompiling on file changes..." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop watching" -ForegroundColor Yellow
}
Write-Host "Auxiliary files will go to: $BuildDir" -ForegroundColor Yellow
Write-Host "PDF will be moved to: $TexDir" -ForegroundColor Yellow

# Compile with latexmk (need to use relative path for outdir)
Push-Location $TexDir
try {
    $RelativeTexFile = [System.IO.Path]::GetFileName($TexFile)
    
    if ($Watch) {
        # Watch mode: use latexmk -pvc (preview continuously)
        # Set up file watcher to move PDF when it changes and clean up log files
        $PdfInBuild = Join-Path $BuildDir "$TexBaseName.pdf"
        $Watcher = New-Object System.IO.FileSystemWatcher
        $Watcher.Path = $BuildDir
        $Watcher.Filter = "$TexBaseName.pdf"
        $Watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite
        
        # Capture variables for script block
        $script:WatchBuildDir = $BuildDir
        $script:WatchTexDir = $TexDir
        $script:WatchTexBaseName = $TexBaseName
        
        $Action = {
            param($Source, $Event)
            Start-Sleep -Milliseconds 1000  # Wait for file to be fully written
            
            # Move PDF from build/ to main directory
            $PdfInBuild = Join-Path $script:WatchBuildDir "$script:WatchTexBaseName.pdf"
            $PdfInMain = Join-Path $script:WatchTexDir "$script:WatchTexBaseName.pdf"
            if (Test-Path $PdfInBuild) {
                Move-Item -Path $PdfInBuild -Destination $PdfInMain -Force -ErrorAction SilentlyContinue
            }
            
            # Move log file from main directory to build/ if it exists there
            $LogInMain = Join-Path $script:WatchTexDir "$script:WatchTexBaseName.log"
            $LogInBuild = Join-Path $script:WatchBuildDir "$script:WatchTexBaseName.log"
            if (Test-Path $LogInMain) {
                Move-Item -Path $LogInMain -Destination $LogInBuild -Force -ErrorAction SilentlyContinue
            }
        }
        
        $OnChanged = Register-ObjectEvent -InputObject $Watcher -EventName "Changed" -Action $Action
        $Watcher.EnableRaisingEvents = $true
        
        Write-Host "`n[WATCH MODE] File watcher started. Saving $RelativeTexFile will trigger recompilation..." -ForegroundColor Cyan
        
        # Run latexmk in preview continuous mode
        try {
            if ($Engine -eq "xelatex") {
                latexmk -pdf -pvc -outdir=build -xelatex $RelativeTexFile
            } elseif ($Engine -eq "lualatex") {
                latexmk -pdf -pvc -outdir=build -lualatex $RelativeTexFile
            } else {
                latexmk -pdf -pvc -outdir=build $RelativeTexFile
            }
        } finally {
            # Final cleanup: move PDF and log file one more time when watch mode ends
            Start-Sleep -Milliseconds 500
            Move-Pdf -BuildDir $BuildDir -TexDir $TexDir -TexBaseName $TexBaseName
            
            $LogInMain = Join-Path $TexDir "$TexBaseName.log"
            $LogInBuild = Join-Path $BuildDir "$TexBaseName.log"
            if (Test-Path $LogInMain) {
                Move-Item -Path $LogInMain -Destination $LogInBuild -Force -ErrorAction SilentlyContinue
            }
            
            # Cleanup watcher when done
            Unregister-Event -SourceIdentifier $OnChanged.Name -ErrorAction SilentlyContinue
            $Watcher.Dispose()
        }
    } else {
        # Normal one-time compilation
        if ($Engine -eq "xelatex") {
            latexmk -pdf -outdir=build -xelatex $RelativeTexFile
        } elseif ($Engine -eq "lualatex") {
            latexmk -pdf -outdir=build -lualatex $RelativeTexFile
        } else {
            latexmk -pdf -outdir=build $RelativeTexFile
        }
        
        # Move PDF after compilation
        Move-Pdf -BuildDir $BuildDir -TexDir $TexDir -TexBaseName $TexBaseName
        
        # Check if compilation was successful
        $PdfInMain = Join-Path $TexDir "$TexBaseName.pdf"
        if (Test-Path $PdfInMain) {
            Write-Host "`n[SUCCESS] Compilation successful!" -ForegroundColor Green
            Write-Host "  PDF: $PdfInMain" -ForegroundColor Green
            Write-Host "  Aux files: $BuildDir" -ForegroundColor Green
        } else {
            $LogFile = Join-Path $BuildDir "$TexBaseName.log"
            Write-Host "`n[ERROR] Compilation failed. Check $LogFile for errors." -ForegroundColor Red
            exit 1
        }
    }
} finally {
    Pop-Location
}

