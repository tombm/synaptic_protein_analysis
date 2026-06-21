
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import analysis_pipeline as ap


st.markdown("# Statistical Analysis")
st.write(
    """
    In this page you can select overall TH values based on the research results.
    Then, apply the TH values you've selected on the entire dataset, run the analysis process and get results.
    """)

images_df = st.session_state['images_df']

if st.session_state['images_df'] is not None:

    images_df = st.session_state['images_df']

    for index, row in images_df.iterrows():
        for channel in ['Pre', 'Post']:
            img = np.array(row[channel])

            mean_brightness = np.mean(img)

            row[f'{channel} results']['mean_brightness'] = mean_brightness

    results_df = images_df[(images_df['Post results'] != {}) & (images_df['Pre results'] != {})]

    try:

        for channel in ['Pre', 'Post']:
            labeled_data = pd.DataFrame.from_dict(st.session_state[f'labeling_{channel}'])

            st.markdown(f"### {channel} labeling")
            st.table(labeled_data)


    except:
        st.error('Save research results first')


    # Analysis parameters (use defaults or could be loaded from saved labeling)
    st.subheader("Analysis Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        contrast_lower = st.slider("Contrast Lower %", 0, 100, 60, key="stat_contrast_lower")
    with col2:
        contrast_upper = st.slider("Contrast Upper %", 0, 100, 0, key="stat_contrast_upper")
    with col3:
        binary_th = st.slider("Binary Threshold", 0, 255, 100, key="stat_binary_th")

    col4, col5, col6, col7 = st.columns(4)
    with col4:
        dots_th = st.slider("Min Dots Size", 0, 30, 10, key="stat_dots_th")
    with col5:
        max_cluster_size = st.slider("Max Cluster Size", 50, 500, 250, key="stat_max_cluster")
    with col6:
        min_samples = st.slider("Min Samples", 5, 100, 35, key="stat_min_samples")
    with col7:
        min_cluster_size = st.slider("Min Cluster Size", 5, 100, 15, key="stat_min_cluster")

    if st.button('Analyze Results'):
        # Get all unique groups from the data
        all_groups = images_df['Group'].unique().tolist()

        # Run analysis on all images with progress bar
        total_images = len(images_df)
        progress_bar = st.progress(0, text="Running analysis...")

        for idx, (index, row) in enumerate(images_df.iterrows()):
            progress_bar.progress((idx + 1) / total_images, text=f"Analyzing image {idx + 1}/{total_images}...")

            for channel in ['Pre', 'Post']:
                img = row[channel]

                # Preprocessing
                lower_th, upper_th = ap.bright_percentile_bounds(img, contrast_lower, contrast_upper)
                contrasted_img = ap.enhance_contrast_pil(img, lower_th, upper_th)
                binary_mask, _ = ap.constant_th(contrasted_img, th=binary_th)
                filtered_bm = ap.remove_small_objects(binary_mask, min_size=dots_th)

                # Clustering
                try:
                    points, vpoints = ap.cluster_staining(
                        filtered_bm,
                        max_cluster_size=max_cluster_size,
                        min_samples=min_samples,
                        min_cluster_size=min_cluster_size
                    )
                    cluster_stats = ap.calculate_cluster_stats(vpoints)
                    row[f'{channel} results']['mean_cluster_size'] = cluster_stats['mean_cluster_size']
                    row[f'{channel} results']['n_clusters'] = cluster_stats['n_clusters']

                    # Calculate mean cluster brightness (brightness of pixels in clusters)
                    if len(vpoints) > 0:
                        img_array = np.array(img)
                        cluster_brightness_values = []
                        for _, point in vpoints.iterrows():
                            x, y = int(point['x']), int(point['y'])
                            if 0 <= y < img_array.shape[0] and 0 <= x < img_array.shape[1]:
                                cluster_brightness_values.append(img_array[y, x])
                        row[f'{channel} results']['mean_cluster_brightness'] = np.mean(cluster_brightness_values) if cluster_brightness_values else 0
                    else:
                        row[f'{channel} results']['mean_cluster_brightness'] = 0
                except Exception as e:
                    row[f'{channel} results']['mean_cluster_size'] = 0
                    row[f'{channel} results']['n_clusters'] = 0
                    row[f'{channel} results']['mean_cluster_brightness'] = 0

        progress_bar.progress(1.0, text="Analysis complete!")
        st.success("Analysis complete!")


        # extract data as CSV button
        csv_bytes = ap.results_df_to_features(images_df).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv_bytes,
            file_name="cluster_features_all_images.csv",
            mime="text/csv",
            key="stats_download_csv",
        )


        st.subheader("Analysis Results")

        # Mean Image Brightness Comparison
        with st.expander("Mean Image Brightness Comparison", expanded=True):
            for channel in ['Pre', 'Post']:
                data = {group: [] for group in all_groups}
                for index, row in images_df.iterrows():
                    mean_val = row[f'{channel} results'].get('mean_brightness')
                    group = row['Group']
                    if mean_val is not None and group in data:
                        data[group].append(mean_val)
                data = {k: v for k, v in data.items() if len(v) > 0}

                if len(data) > 0:
                    fig = ap.create_boxplot(
                        data,
                        title=f"{channel} - Mean Image Brightness",
                        ylabel="Mean Image Brightness",
                        figsize=(4, 4.8)
                    )
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                else:
                    st.warning(f"No data available for {channel} channel")

        # Mean Cluster Brightness Comparison
        with st.expander("Mean Cluster Brightness Comparison", expanded=True):
            for channel in ['Pre', 'Post']:
                data = {group: [] for group in all_groups}
                for index, row in images_df.iterrows():
                    mean_val = row[f'{channel} results'].get('mean_cluster_brightness')
                    group = row['Group']
                    if mean_val is not None and mean_val > 0 and group in data:
                        data[group].append(mean_val)
                data = {k: v for k, v in data.items() if len(v) > 0}

                if len(data) > 0:
                    fig = ap.create_boxplot(
                        data,
                        title=f"{channel} - Mean Cluster Brightness",
                        ylabel="Mean Cluster Brightness",
                        figsize=(4, 4.8)
                    )
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                else:
                    st.warning(f"No cluster brightness data available for {channel} channel")

        # Mean Cluster Size Comparison
        with st.expander("Mean Cluster Size Comparison", expanded=True):
            for channel in ['Pre', 'Post']:
                data = {group: [] for group in all_groups}
                for index, row in images_df.iterrows():
                    mean_val = row[f'{channel} results'].get('mean_cluster_size')
                    group = row['Group']
                    if mean_val is not None and mean_val > 0 and group in data:
                        data[group].append(mean_val)
                data = {k: v for k, v in data.items() if len(v) > 0}

                if len(data) > 0:
                    fig = ap.create_boxplot(
                        data,
                        title=f"{channel} - Mean Cluster Size",
                        ylabel="Mean Cluster Size (points)",
                        figsize=(4, 4.8)
                    )
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                else:
                    st.warning(f"No cluster data available for {channel} channel")

        # Intensity Distribution
        with st.expander("Intensity Distribution", expanded=True):
            for channel in ['Pre', 'Post']:
                # Collect images per group
                data = {group: [] for group in all_groups}
                for index, row in images_df.iterrows():
                    img = row[channel]
                    group = row['Group']
                    if img is not None and group in data:
                        data[group].append(np.array(img))
                data = {k: v for k, v in data.items() if len(v) > 0}

                if len(data) > 0:
                    fig = ap.create_intensity_distribution(
                        data,
                        title=f"{channel} - Intensity Distribution",
                        figsize=(8, 4)
                    )
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                else:
                    st.warning(f"No intensity data available for {channel} channel")


else:
    st.error('No file uploaded')



