import streamlit as st
import pandas as pd
import analysis_pipeline as ap
import importlib

importlib.reload(ap)


metadata_columns = ['Filename', 'Image name', 'Group', 'Area', 'Pre', 'Post', 'Pre results', 'Post results']

st.set_page_config(page_title="Synaptic Protein Analysis")

st.write("# Synaptic Protein Analysis")

st.sidebar.write("Select a mode above.")

st.markdown(
    """
    This is an interface designed to provide accessible synaptic protein analysis pipeline based on data science tools.
    ### The pipeline:
    - Gallery: Displays all images with filters to provide a broad understanding of the data and edge cases.
    - Research: Allows for the fine-tuning and adjustment of pipeline parameters on a per-image basis.
    - Analysis: Enables the extensive execution of the chosen parameters across all images to generate broad results, including graphs, tables, and statistics.
    
    This tool was developed in the laboratory of Uri Ashery by Tom Ben-Mor.  
    """)

st.write("### Upload file")

uploaded_files = st.file_uploader(
    "Upload .lif confocal image files",
    type="lif",
    accept_multiple_files=True,
    key="standard_uploader")


# supporting S-H pairs constructed from 2 different files
st.write("### Upload S-H paired files")

paired_uploaded_files = st.file_uploader(
    "Optional - Upload S-H paired .lif files",
    type="lif",
    accept_multiple_files=True,
    key="paired_uploader")

paired_metadata = []
if paired_uploaded_files:
    paired_metadata, skipped = ap.extract_paired_metadata(paired_uploaded_files)
    for filename, img_name in skipped:
        st.warning(f"Skipping '{img_name}' in '{filename}': not in 'AREA S/H' format")



st.session_state['images_df'] = None

dfs = []

if uploaded_files:
    lif_files = []
    for file in uploaded_files:
        lif = ap.read_lif(file)
        lif_files.append((lif, file.name))
    dfs.append(ap.create_images_df_mult_files(lif_files))

if paired_metadata:
    paired_df, skipped_pairs = ap.create_images_df_paired(paired_metadata)
    for filename, region, reason in skipped_pairs:
        st.warning(f"Skipping region '{region}' in '{filename}': {reason}")
    dfs.append(paired_df)

if dfs:
    images_df = pd.concat(dfs, ignore_index=True)
    st.session_state['images_df'] = images_df
    st.session_state['labeling'] = {}
