# IRD_spatial_analysis

This repository contains the analysis scripts and notebooks for the analysis of Xenium spatial transcriptomics dataset of the IRD project. This project is a follow-up to a clinical trial to understand the microenvironmental changes following autologous stem cell transplant (ASCT) treatment for multiple myeloma. To achieve this, spatial mRNA profiling (10x Xenium) was performed on patient bone marrow biopsy samples as well as normal donor samples.

## Repository structure

- **IRD_sketch_annotation.ipynb** - Cell type annotation using Seurat sketching and unsupervised clustering (R)
- **Spatial_analysis_IRD.ipynb** - Select downstream spatial analysis including cell composition, gene expression, immune microenvironment characterization, and neighborhood identification (Python)
- **utils/** - Helper functions directory containing:
  - `plot_utils.py` - Plotting functions for statistical visualizations in Python
  - `spatial_utils.py` - Spatial analysis utilities in Python (Delaunay triangulation, neighbor detection)