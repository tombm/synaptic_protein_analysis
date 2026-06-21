
import streamlit as st
import matplotlib.pyplot as plt


st.markdown("# Image Gallery")
st.write("In this page you can see all images from the file, in their original form.")

images_df = st.session_state['images_df']

if st.session_state['images_df'] is not None:
    filenames = st.session_state['images_df']['Filename'].unique()

    for filename in filenames:
        with st.expander(f"📁 {filename}", expanded=True):
            file_images = st.session_state['images_df'][
                st.session_state['images_df']['Filename'] == filename]

            for index, row in file_images.iterrows():
                st.markdown(f"**{row['Image name']}** — Area: {row['Area']}")
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))

                for i, staining in enumerate(['Pre', 'Post']):
                    image_data = row[f'{staining}']
                    axes[i].imshow(image_data, cmap='gray')
                    axes[i].set_title(f'{staining} Synaptic Image', fontsize=10)
                    axes[i].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

else:
    st.error('No file uploaded')



