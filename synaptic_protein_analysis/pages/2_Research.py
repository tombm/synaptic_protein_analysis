
import streamlit as st
import matplotlib.pyplot as plt
import analysis_pipeline as ap
from analysis_pipeline import style_df


st.markdown(
    """
    # Research
    ### Parameter Adjustments

    In this page you can select an image, and adjust it's parameters individually for each stage of the analysis pipeline.
    ### Pipeline Stages:
    - Contrast Enhancement
    - Binary Mask
    - Clustering
    """
)

if st.session_state['images_df'] is not None:

    images_df = st.session_state['images_df']

    # 1 - IMAGES LIST
    st.subheader("Images Information")

    with st.expander("Images Table", expanded=True):
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_filenames = st.multiselect("Filter by Filename", options=images_df['Filename'].unique())
        with col2:
            selected_groups = st.multiselect("Filter by Group", options=images_df['Group'].unique())
        with col3:
            selected_areas = st.multiselect("Filter by Area", options=images_df['Area'].unique())

        # Apply filters
        filtered_df = images_df.copy()
        if selected_filenames:
            filtered_df = filtered_df[filtered_df['Filename'].isin(selected_filenames)]
        if selected_groups:
            filtered_df = filtered_df[filtered_df['Group'].isin(selected_groups)]
        if selected_areas:
            filtered_df = filtered_df[filtered_df['Area'].isin(selected_areas)]

        st.write(style_df(filtered_df[['Filename', 'Image name', 'Group', 'Area']]))

    # 2 - SELECT IMAGE
    st.subheader("Select Image")

    col1, col2 = st.columns(2)
    with col1:
        selected_file = st.selectbox("Select file", options=images_df['Filename'].unique())
    file_filtered_df = images_df[images_df['Filename'] == selected_file]

    with col2:
        unique_values = file_filtered_df['Image name'].unique()
        selected_image = st.selectbox("Select image", unique_values)

    filtered_row = file_filtered_df[file_filtered_df['Image name'] == selected_image].iloc[0]

    row_display = filtered_row[['Filename', 'Image name', 'Group', 'Area']].to_frame().T
    row_display.index.name = 'Index'
    st.write(row_display)

    # SHOW ORIGINAL PRE & POST
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    for i, staining in enumerate(['Pre', 'Post']):
        image_data = filtered_row[f'{staining}']
        axes[i].imshow(image_data, cmap='gray')
        axes[i].set_title(f'{staining} Synaptic Image', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 4 - Pre-Processing
    # Binary mask, contrast enhancement, remove dots
    st.subheader("Image Pre-Processing (Noise Reduction)")


    selected_channel = st.selectbox("Select channel:", ['Pre', 'Post'])
    img = filtered_row[f'{selected_channel}']

    # CONTRAST

    st.markdown("""**Contrast Enhancement**""")
    default_lower_percentile, default_upper_percentile = [60, 0]

    selected_lower = st.slider(
        'Choose **Lower** Percentile (clip darkest)',
        min_value=0,
        max_value=100,
        value=default_lower_percentile,
        key='lower_th_slider'
    )

    selected_upper = st.slider(
        'Choose **Upper** Percentile (clip brightest)',
        min_value=0,
        max_value=100,
        value=default_upper_percentile,
        key='upper_th_slider'
    )

    lower_percentile = selected_lower
    upper_percentile = selected_upper

    lower_th, upper_th = ap.bright_percentile_bounds(img, lower_percentile, upper_percentile)
    contrasted_img = ap.enhance_contrast_pil(img, lower_th, upper_th)

    fig = ap.view_two_images(img, contrasted_img, f'{selected_channel} - Contrast Enhancement',
                             'Original', 'Contrasted')
    st.pyplot(fig)

    # BINARY MASK
    st.write('Binary Mask')
    default_th = 100

    selected_th = st.slider(
        'Choose binary mask threshold:',
        min_value=0,
        max_value=255,
        value=default_th,
        key='th_slider'
    )

    binary_mask, per_above_hundred = ap.constant_th(contrasted_img, th=selected_th)

    fig = ap.view_two_images(contrasted_img, binary_mask, f'{selected_channel} - Binary Mask',
                             'Contrasted', 'Binary Mask')
    st.pyplot(fig)


    # REMOVE DOTS
    st.write('\n**Remove Dots**')
    default_dots_th = 10

    selected_dots_th = st.slider(
        'Choose a max size for small dots removal:',
        min_value=0,
        max_value=30,
        value=default_dots_th,
        key='dots_removal_slider'
    )

    filtered_bm = ap.remove_small_objects(binary_mask, min_size=selected_dots_th)

    fig = ap.view_two_images(binary_mask, filtered_bm, f'{selected_channel} - Remove Dots',
                             'Binary Mask', 'Removed Dots')
    st.pyplot(fig)

    # CLUSTERING PARAMETERS
    st.write('\n**Clustering Parameters**')
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_max_cluster_size = st.slider(
            'Max cluster size',
            min_value=20, max_value=350, value=100, key='max_cluster_size_slider'
        )
    with col2:
        selected_min_samples = st.slider(
            'Min samples',
            min_value=5, max_value=100, value=35, key='min_samples_slider'
        )
    with col3:
        selected_min_cluster_size = st.slider(
            'Min cluster size',
            min_value=5, max_value=100, value=15, key='min_cluster_size_slider'
        )

    if st.button('Cluster'):
        print('Clustering...')
        try:
            points, vpoints = ap.cluster_staining(
                filtered_bm,
                max_cluster_size=selected_max_cluster_size,
                min_samples=selected_min_samples,
                min_cluster_size=selected_min_cluster_size
            )
            print('Plotting Clusters...')
            title = f'{selected_channel} - Clustering'
            fig = ap.view_clustering(title, filtered_bm, vpoints, 'Filtered dots', 'Clusters Colored')
            st.pyplot(fig)
        except Exception as e:
            st.error(e)


    for channel in ('Pre', 'Post'):
        if f'labeling_{channel}' not in st.session_state:
            st.session_state[f'labeling_{channel}'] = {
                "Filename": [],
                "Image": [],
                "Contrast Lower TH": [],
                "Contrast Upper TH": [],
                "Binary Mask TH": [],
                "Dots TH": [],
                "Max Cluster Size": [],
                "Min Samples": [],
                "Min Cluster Size": []
            }

    if st.button('Save Results'):
        st.session_state[f'labeling_{selected_channel}']["Filename"].append(selected_file)
        st.session_state[f'labeling_{selected_channel}']["Image"].append(selected_image)
        st.session_state[f'labeling_{selected_channel}']["Contrast Lower TH"].append(selected_lower)
        st.session_state[f'labeling_{selected_channel}']["Contrast Upper TH"].append(selected_upper)
        st.session_state[f'labeling_{selected_channel}']["Binary Mask TH"].append(selected_th)
        st.session_state[f'labeling_{selected_channel}']["Dots TH"].append(selected_dots_th)
        st.session_state[f'labeling_{selected_channel}']["Max Cluster Size"].append(selected_max_cluster_size)
        st.session_state[f'labeling_{selected_channel}']["Min Samples"].append(selected_min_samples)
        st.session_state[f'labeling_{selected_channel}']["Min Cluster Size"].append(selected_min_cluster_size)
        st.success("Results saved!")


else:
    st.error('No file uploaded')


