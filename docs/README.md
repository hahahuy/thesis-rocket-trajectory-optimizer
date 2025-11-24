# Documentation Structure

This directory contains comprehensive documentation for the Rocket Trajectory Optimizer project. This guide explains the organization and purpose of each document.

## Quick Navigation

### Getting Started
- **[SETUP.md](SETUP.md)**: Main setup entry point with quick start guide
- **[SETUP_ENVIRONMENT.md](SETUP_ENVIRONMENT.md)**: Complete environment setup (Python, C++, Windows, CI/CD)
- **[SETUP_HSL_MUMPS.md](SETUP_HSL_MUMPS.md)**: HSL/MUMPS linear solver setup for IPOPT

### Architecture & Design
- **[DESIGN.md](DESIGN.md)**: High-level system architecture and design overview
- **[architecture_diagram.md](architecture_diagram.md)**: Mermaid diagrams for all PINN architectures
- **[ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md)**: PINN architecture evolution history

### Work Package Documentation
- **[wp1_comprehensive_description.md](wp1_comprehensive_description.md)**: Physics core (6-DOF dynamics library)
- **[wp2_comprehensive_description.md](wp2_comprehensive_description.md)**: Optimal control baseline (CasADi/IPOPT solver)
- **[wp3_comprehensive_description.md](wp3_comprehensive_description.md)**: Dataset generation and preprocessing
- **[wp4_comprehensive_description.md](wp4_comprehensive_description.md)**: PINN training and models

### Results & Analysis
- **[RESULTS_AND_VALIDATION.md](RESULTS_AND_VALIDATION.md)**: Validation results (WP1-4) and experiment summaries (exp1-5)
- **[expANAL_SOLS.md](expANAL_SOLS.md)**: C3 architecture implementation guide and RMSE analysis

### Research & Development
- **[thesis_notes.md](thesis_notes.md)**: Research notes and model development decisions
- **[Contributing.md](Contributing.md)**: Development guidelines and contribution process

### Specialized Topics
- **[UI_ROCKET_SHAPE_INTEGRATION.md](UI_ROCKET_SHAPE_INTEGRATION.md)**: UI integration documentation

---

## Documentation Organization

### Setup Documentation (`SETUP_*.md`)

Setup guides are organized in a logical sequence:

1. **SETUP.md**: Main entry point with quick start and prerequisites
2. **SETUP_ENVIRONMENT.md**: Complete environment setup including:
   - Python/Conda environment
   - Core C++ libraries (Eigen, LibTorch, HDF5, etc.)
   - CMake configuration
   - Windows-specific instructions
   - CI/CD pipeline setup
3. **SETUP_HSL_MUMPS.md**: HSL/MUMPS linear solver setup for IPOPT

### Design Documentation

- **DESIGN.md**: Navigation hub with inline references to WP comprehensive docs
  - High-level architecture overview
  - State/control representations
  - Physics and dynamics
  - Constraints and integration
  - References to detailed WP documentation

- **architecture_diagram.md**: Visual architecture diagrams using Mermaid
  - Baseline PINN
  - Direction A (Latent ODE)
  - Direction B (Sequence/Transformer)
  - Direction C (Hybrid)
  - Direction C1 (Enhanced Hybrid)
  - Direction C2 (Shared Stem + Dedicated Branches)
  - Direction C3 (Enhanced C2 with solutions)

- **ARCHITECTURE_CHANGELOG.md**: Complete history of PINN architecture evolution
  - Original baseline model
  - All architecture directions (A, B, C, C1, C2)
  - Implementation details, configuration, status

### Work Package Comprehensive Descriptions

Each WP has a comprehensive description document covering:

- Executive Summary
- Technical Overview
- Implementation Architecture
- Development Journey
- Testing & Validation Framework
- Current Status & Results
- Lessons Learned
- Future Recommendations
- Operations: How to Run

**Files**:
- `wp1_comprehensive_description.md`: Physics core
- `wp2_comprehensive_description.md`: OCP solver
- `wp3_comprehensive_description.md`: Dataset generation
- `wp4_comprehensive_description.md`: PINN training

### Results & Validation

- **RESULTS_AND_VALIDATION.md**: Consolidated validation and experiment results
  - WP1 validation results
  - WP2 validation results
  - WP3 validation results
  - WP4 validation results
  - Experiment summaries (exp1-5)
  - Cross-WP integration testing

- **expANAL_SOLS.md**: C3 architecture implementation guide
  - Problem analysis and root causes
  - C2 vs C3 comparison
  - Solution implementation details
  - Integration guide
  - Expected performance

### Research Documentation

- **thesis_notes.md**: Research notes and development decisions
  - Model development stages
  - Architecture directions (Base, A, B, C, C1, C2)
  - Data flow and design decisions
  - Pros and cons analysis

- **Contributing.md**: Development guidelines
  - Code style (C++ and Python)
  - Pull request process
  - Testing requirements
  - Documentation standards

---

## Reading Guide

### For New Contributors

1. Start with **[SETUP.md](SETUP.md)** to set up your environment
2. Read **[DESIGN.md](DESIGN.md)** for system architecture overview
3. Review relevant **WP comprehensive descriptions** for your area of work
4. Check **[Contributing.md](Contributing.md)** for development guidelines

### For Researchers

1. Read **[thesis_notes.md](thesis_notes.md)** for research context
2. Review **[ARCHITECTURE_CHANGELOG.md](ARCHITECTURE_CHANGELOG.md)** for model evolution
3. Check **[architecture_diagram.md](architecture_diagram.md)** for visual architecture
4. See **[RESULTS_AND_VALIDATION.md](RESULTS_AND_VALIDATION.md)** for experimental results

### For Users

1. Follow **[SETUP.md](SETUP.md)** for installation
2. Check **WP comprehensive descriptions** for operations guides:
   - [WP1 Operations](wp1_comprehensive_description.md#wp1-operations-how-to-run)
   - [WP2 Operations](wp2_comprehensive_description.md#wp2-operations-how-to-run)
   - [WP3 Operations](wp3_comprehensive_description.md#wp3-operations-how-to-run)
   - [WP4 Usage](wp4_comprehensive_description.md#usage)

---

## File Naming Conventions

- **SETUP_*.md**: Setup and installation guides
- **wp*_comprehensive_description.md**: Work package detailed documentation
- **DESIGN.md**: High-level design navigation
- **ARCHITECTURE_*.md**: Architecture and model evolution
- **RESULTS_AND_VALIDATION.md**: Testing, validation, and experiment results
- **Contributing.md**: Development guidelines

---

## Document Maintenance

When updating documentation:

1. **Architecture changes**: Update `ARCHITECTURE_CHANGELOG.md` and `architecture_diagram.md`
2. **Setup changes**: Update relevant `SETUP_*.md` files
3. **WP changes**: Update corresponding `wp*_comprehensive_description.md`
4. **Design changes**: Update `DESIGN.md` with inline references
5. **New experiments**: Add summaries to `RESULTS_AND_VALIDATION.md`

---

## Questions?

- Check the relevant comprehensive description for detailed information
- Review [DESIGN.md](DESIGN.md) for architecture questions
- See [SETUP.md](SETUP.md) for setup issues
- Open an issue on GitHub for questions or suggestions
