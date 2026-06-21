# Synaptic Protein Analysis

An interactive Streamlit application for analyzing synaptic protein staining
in confocal microscopy images. It provides an accessible, data-science-based
pipeline for processing `.lif` image files and comparing results across groups.

Developed in the laboratory of Uri Ashery by Tom Ben-Mor.

## Features

The app is organized into pages:

- **Home** – Upload `.lif` confocal image files (including optional S–H paired files).
- **Gallery** – View all uploaded images (Pre and Post synaptic channels) in their original form.
- **Research** – Select a single image and fine-tune pipeline parameters per stage:
  contrast enhancement, binary mask, dot removal, and clustering.
- **Statistics** – Apply chosen parameters across the whole dataset and generate
  comparison plots (mean brightness, cluster size, cluster brightness, intensity
  distribution), with the option to download results as CSV.

## How to Run

1. Clone the repository:

```
git clone git@github.com:USERNAME/synaptic_protein_analysis.git
cd synaptic_protein_analysis
```

2. Install the dependencies:

```
pip install -r requirements.txt
```

3. Start the app:

```
streamlit run Hello.py
```

This launches the app and opens it in your default web browser. Upload your
`.lif` files on the Home page to begin.

## Project structure

```
synaptic_protein_analysis/
├── Hello.py
├── analysis_pipeline.py
├── requirements.txt
└── pages/
    ├── 1_Gallery.py
    ├── 2_Research.py
    └── 3_Statistics.py
```

## Usage

1. Upload one or more `.lif` files on the Home page.
2. Browse them in the Gallery to get an overview.
3. Use Research to tune parameters on individual images.
4. Run the full analysis in Statistics and download the results.
