
import cv2
import numpy as np
import seaborn as sns
from readlif.reader import LifFile
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import hdbscan

from scipy.stats import ttest_ind


areas = ['DG', 'SUB', 'MF', 'CA1', 'CA3', 'SSP']
metadata_columns = ['Filename', 'Image name', 'Group', 'Area', 'Pre', 'Post', 'Pre results', 'Post results']

PALETTE = {
    "pre_5x":  {"face": "#AD4DCF", "edge": "#6B4FC6", "dot": "#6B4FC6"},
    "pre_wt":  {"face": "#96969A", "edge": "#7A7A7A", "dot": "#7A7A7A"},
    "post_5x": {"face": "#590C77", "edge": "#4B30B8", "dot": "#4B30B8"},
    "post_wt": {"face": "#555555", "edge": "#5F5F5F", "dot": "#5F5F5F"},
}

S_H_TO_CHANNEL = {"S": "Pre", "H": "Post"}



def read_lif(img_path):
    lif = LifFile(img_path)
    return lif

# 1 - create DF
def create_images_df(lif):
    group = ''
    imgs_metadata = []
    img_list = [img for img in lif.get_iter_image()]

    for i, img in enumerate(img_list):
        img_name = img.name

        if ('5X' in img_name) or ('5x' in img_name):
            group = '5x'
        elif 'wt' in img_name.lower():
            group = 'wt'

        if (i <= len(areas) - 1) and group is not None:
            area = areas[i]

            ch_imgs = [img.get_frame(c=ch) for ch in range(img.channels)]
            pre = ch_imgs[3]
            post = ch_imgs[2]

            img_data = [img_name, img_name, group, area, pre, post, {}, {}]
            imgs_metadata.append(img_data)

    images_df = pd.DataFrame(imgs_metadata, columns=metadata_columns)
    return images_df

def create_images_df_mult_files(lif_files: list) -> pd.DataFrame:
    """
    Process multiple LIF files and combine into a single DataFrame.
    Args: lif_files: List of tuples (lif_object, filename) for each uploaded file
    Returns: Combined DataFrame with images from all files
    """
    imgs_metadata = []

    for lif, filename in lif_files:
        group = ''
        if '5x' in filename.lower():
            group = '5x'
        elif 'wt' in filename.lower():
            group = 'wt'
        if (group and 'hbot' in filename.lower()):
            group += ' hbot'

        img_list = [img for img in lif.get_iter_image()]

        for i, img in enumerate(img_list):
            img_name = img.name

            if (i <= len(areas) - 1) and group is not None:
                area = areas[i]

                ch_imgs = [img.get_frame(c=ch) for ch in range(img.channels)]
                pre = ch_imgs[3]
                post = ch_imgs[2]

                img_data = [filename, img_name, group, area, pre, post, {}, {}]
                imgs_metadata.append(img_data)

    images_df = pd.DataFrame(imgs_metadata, columns=metadata_columns)
    return images_df


# TODO S-H
def extract_paired_metadata(uploaded_files):
    """
    Parse image names from S-H paired LIF files.
    Expected name format: "AREA S" or "AREA H" (e.g., "CA1 H", "DG S").

    Returns:
        paired_metadata: list of dicts with filename, img_name, region, channel, img
        skipped: list of (filename, img_name) tuples for names that didn't match the format
    """
    paired_metadata = []
    skipped = []

    for file in uploaded_files:
        lif = read_lif(file)
        for img in lif.get_iter_image():
            img_name = img.name
            parts = img_name.strip().split()

            if len(parts) != 2 or parts[0] not in areas or parts[1] not in S_H_TO_CHANNEL:
                skipped.append((file.name, img_name))
                continue
            region = parts[0]
            channel = S_H_TO_CHANNEL[parts[1]]
            paired_metadata.append({
                "filename": file.name,
                "img_name": img_name,
                "region": region,
                "channel": channel,
                "img": img,
            })

    return paired_metadata, skipped

# TODO S-H
def create_images_df_paired(paired_metadata):
    """
    Build a DataFrame from S-H paired metadata.
    Groups entries by (filename, region) and creates one row per valid pair,
    where Pre comes from the S image and Post comes from the H image.

    Returns:
        images_df: DataFrame in the same format as create_images_df_mult_files
        skipped_pairs: list of (filename, region, reason) for invalid groupings
    """
    # Group by (filename, region)
    groups = {}
    for entry in paired_metadata:
        key = (entry["filename"], entry["region"])
        groups.setdefault(key, []).append(entry)

    imgs_metadata = []
    skipped_pairs = []

    for (filename, region), entries in groups.items():
        if len(entries) != 2:
            skipped_pairs.append((filename, region, f"expected 2 images, found {len(entries)}"))
            continue

        s_entry = next((e for e in entries if e["channel"] == S_H_TO_CHANNEL["S"]), None)
        h_entry = next((e for e in entries if e["channel"] == S_H_TO_CHANNEL["H"]), None)

        if s_entry is None or h_entry is None:
            skipped_pairs.append((filename, region, "need exactly one S and one H image"))
            continue

        # Determine group from filename (same logic as create_images_df_mult_files)
        group = ''
        if '5x' in filename.lower():
            group = '5x'
        elif 'wt' in filename.lower():
            group = 'wt'
        if group and 'hbot' in filename.lower():
            group += ' hbot'

        # Pre channel from S image, Post channel from H image
        s_img = s_entry["img"]
        h_img = h_entry["img"]

        s_ch_imgs = [s_img.get_frame(c=ch) for ch in range(s_img.channels)]
        h_ch_imgs = [h_img.get_frame(c=ch) for ch in range(h_img.channels)]

        pre = s_ch_imgs[3]
        post = h_ch_imgs[2]

        img_name = f"{region} (S+H)"
        img_data = [filename, img_name, group, region, pre, post, {}, {}]
        imgs_metadata.append(img_data)

    images_df = pd.DataFrame(imgs_metadata, columns=metadata_columns)
    return images_df, skipped_pairs


# 2 - style DF
def style_df(df):
    styled = df.style.set_properties(
        **{"background-color": "#2C2F3A", "color": "#E5E7EB", "border-color": "#111827"}
    )
    return styled


# 3 - PRE-PROCESSING
# contrast enhancement
def bright_percentile_bounds(img: np.ndarray, lower_percentile: float, upper_percentile_clip: float,
                             ignore_nan: bool = True):

    a = np.asarray(img).ravel()

    # Calculate the upper percentile value (100 - clip_percentage)
    upper_percentile = 100 - upper_percentile_clip

    if ignore_nan:
        lower_th = float(np.nanpercentile(a, lower_percentile))
        upper_th = float(np.nanpercentile(a, upper_percentile))
    else:
        lower_th = float(np.percentile(a, lower_percentile))
        upper_th = float(np.percentile(a, upper_percentile))

    return lower_th, upper_th

def enhance_contrast_pil(img: Image.Image, clip_low: float, clip_high: float) -> Image.Image:
    """
    Contrast-stretch a PIL image.

    - Converts to grayscale ('L') if needed.
    - clip_low/clip_high are intensity bounds in the image's scale (0..255 after conversion).
    - Returns a new PIL Image in 'L' mode.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")

    # Ensure 2D by converting to grayscale
    if img.mode != "L":
        img = img.convert("L")

    if clip_low >= clip_high:
        raise ValueError("clip_low must be less than clip_high")

    arr = np.asarray(img, dtype=np.float32)

    # Apply contrast stretch
    stretched = np.clip(arr, clip_low, clip_high)
    stretched = (stretched - clip_low) / (clip_high - clip_low + 1e-8) * 255.0
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    return Image.fromarray(stretched, mode="L")


# binary mask
def constant_th(image, th=200, count_above=100):
    # the th is the tf % of the max brightness
    total_pixels = 0
    count_above_hundred = 0

    image = np.asarray(image, dtype=np.float32)

    binary_mask = image.copy()
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            total_pixels += 1
            if binary_mask[i][j] >= th:
                binary_mask[i][j] = 255
            if binary_mask[i][j] >= count_above:
                count_above_hundred += 1
            else:
                binary_mask[i][j] = 0

    per_above_hundred = (count_above_hundred / total_pixels) * 100
    return binary_mask, per_above_hundred


# remove dots
def remove_small_objects(image, min_size: int = 30) -> np.ndarray:
    # gets binary masks, removes small dots and returns result

    if isinstance(image, Image.Image):
        img_array = np.asarray(image, dtype=np.float32)
    else:
        img_array = np.asarray(image, dtype=np.float32)

    binary_mask = np.asarray(img_array, dtype=np.uint8)

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1

    filtered = np.zeros_like(binary_mask)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered[output == i + 1] = 255

    return filtered


# clustering
def cluster_staining(img, max_cluster_size=250, min_samples=35, min_cluster_size=15, cluster_selection_epsilon=0.0):
    # If mask is empty (all 0), return DataFrames with zeros
    if isinstance(img, Image.Image):
        image = np.asarray(img, dtype=np.float32)
    else:
        image = np.asarray(img, dtype=np.float32)

    if np.count_nonzero(image) == 0:
        empty_points = pd.DataFrame([{'x': 0, 'y': 0, 'label': 0}])
        empty_vpoints = pd.DataFrame([{'x': 0, 'y': 0, 'label': 0, 'color': (0, 0, 0)}])
        return empty_points, empty_vpoints

    coords = np.argwhere(image == 255)
    points = pd.DataFrame({'x': coords[:, 1], 'y': coords[:, 0]})

    if points.shape[0] > 0.5 * (img.shape[0] * img.shape[1]):
        raise ValueError('Parameters are too saturated')
    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    clusterer.fit(points.to_numpy())

    points['label'] = clusterer.labels_

    vpoints = points[points['label'] != -1].reset_index(drop=True)

    vpoints['color'] = vpoints['label'].apply(
        lambda x: sns.color_palette()[x % len(sns.color_palette())]
    )

    return points, vpoints


def calculate_cluster_stats(vpoints):
    """
    Calculate cluster statistics from clustering results.

    Args:
        vpoints: DataFrame with clustered points (output from cluster_staining)

    Returns:
        dict with cluster statistics including mean_cluster_size, n_clusters, total_points
    """
    if vpoints is None or len(vpoints) == 0 or vpoints['label'].iloc[0] == 0:
        return {
            'mean_cluster_size': 0,
            'n_clusters': 0,
            'total_points': 0,
            'cluster_sizes': []
        }

    cluster_sizes = vpoints['label'].value_counts().values
    n_clusters = len(cluster_sizes)
    mean_cluster_size = np.mean(cluster_sizes) if n_clusters > 0 else 0

    return {
        'mean_cluster_size': float(mean_cluster_size),
        'n_clusters': int(n_clusters),
        'total_points': int(len(vpoints)),
        'cluster_sizes': cluster_sizes.tolist()
    }


def results_df_to_features(images_df):
    """Flatten the Pre/Post results dicts into one tidy row per image for CSV export."""
    feature_keys = ['mean_brightness', 'mean_cluster_brightness', 'mean_cluster_size', 'n_clusters']
    records = []
    for _, row in images_df.iterrows():
        rec = {k: row[k] for k in ['Filename', 'Image name', 'Group', 'Area']}
        for channel in ['Pre', 'Post']:
            results = row[f'{channel} results'] or {}
            for key in feature_keys:
                rec[f'{channel} {key}'] = results.get(key)
        records.append(rec)
    return pd.DataFrame(records)


# UTILS
def view_two_images(img1, img2, title, name1, name2):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.title(f'{name1}')
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'{name2}')
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    #plt.show()
    return fig



def view_clustering(title, filtered_bm, vpoints, name1, name2):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.title(name1)
    plt.imshow(filtered_bm, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(name2)
    plt.scatter(vpoints['x'], vpoints['y'], s=1, c=vpoints['color'])
    plt.xlim(0, filtered_bm.shape[1])
    plt.ylim(filtered_bm.shape[0], 0)
    plt.gca().set_aspect('equal')  # match pixel ratio
    plt.axis('off')

    #plt.show()
    return fig


#
# # BOXPLOT MEAN BRIGHT
# Assuming your PALETTE is defined globally or passed in
DEFAULT_PALETTE = {
    "5x": {"face": "#ff9999", "edge": "#cc0000", "dot": "#ff0000"},
    "wt": {"face": "#99ff99", "edge": "#00cc00", "dot": "#00ff00"},
    "5x hbot": {"face": "#ffcc99", "edge": "#cc6600", "dot": "#ff8800"},
    "wt hbot": {"face": "#99ccff", "edge": "#0066cc", "dot": "#0088ff"}
}


def st_boxplot_comparison(df, stat_col, group_col="Group", title="Comparison", palette=None):
    """
    Creates a clean, publication-ready boxplot with jittered dots and p-values.
    Designed for Streamlit.
    """
    palette = palette or DEFAULT_PALETTE

    # 1. Setup Data
    groups = df[group_col].unique()
    data_to_plot = [df[df[group_col] == g][stat_col].dropna().values for g in groups]

    # 2. Setup Figure
    # Using a slightly smaller font size globally for a "clean" look
    plt.rc('font', size=9)
    fig, ax = plt.subplots(figsize=(5, 6))

    # 3. Draw Boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False, widths=0.6)

    for i, group_name in enumerate(groups):
        color_key = group_name.lower()  # normalization
        style = palette.get(color_key, palette.get("wt"))  # fallback

        # Style Boxes
        bp["boxes"][i].set(facecolor=style["face"], edgecolor=style["edge"], linewidth=1.5)

        # Style Whiskers/Caps/Medians
        for part in ["whiskers", "caps"]:
            plt.setp(bp[part][i * 2:(i * 2) + 2], color="#6b6b6b", linewidth=1.1)
        plt.setp(bp["medians"][i], color="#222222", linewidth=2)

        # 4. Add Jittered Dots
        rng = np.random.default_rng(42)
        y = data_to_plot[i]
        x = rng.normal(loc=i + 1, scale=0.08, size=len(y))
        ax.scatter(x, y, s=30, alpha=0.7, color=style["dot"],
                   edgecolors="black", linewidths=0.5, zorder=3)

    # 5. Stats & Brackets (Welch's T-Test)
    if len(data_to_plot) == 2:
        stat_val, p = ttest_ind(data_to_plot[0], data_to_plot[1], equal_var=False)
        if p < 0.05:
            y_max = max([arr.max() for arr in data_to_plot])
            y_range = y_max - min([arr.min() for arr in data_to_plot])

            # Bracket coords
            y_bracket = y_max + (y_range * 0.1)
            h = y_range * 0.05
            ax.plot([1, 1, 2, 2], [y_bracket, y_bracket + h, y_bracket + h, y_bracket], lw=1.2, c="black")

            # Stars
            stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*"
            ax.text(1.5, y_bracket + h, stars, ha="center", va="bottom", fontsize=12)
            ax.set_ylim(None, y_bracket + (y_range * 0.3))  # Make room for bracket

    # 6. Styling
    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xticklabels([g.upper() for g in groups])
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    # 7. Render in Streamlit
    return fig


def create_boxplot(data: dict, title="Comparison", ylabel="Value", palette=None, figsize=(5, 6)):
    """
    Creates a publication-ready boxplot with jittered dots and p-values.

    Args:
        data: Dictionary with group names as keys and lists of values as values.
              Example: {"5x": [1.2, 1.5, 1.3], "wt": [0.8, 0.9, 0.7]}
        title: Plot title
        ylabel: Y-axis label
        palette: Optional color palette dict. If None, uses DEFAULT_PALETTE.
        figsize: Tuple of (width, height) for figure size.

    Returns:
        matplotlib figure
    """
    palette = palette or DEFAULT_PALETTE

    groups = list(data.keys())
    data_to_plot = [np.array(data[g], dtype=float) for g in groups]
    data_to_plot = [arr[~np.isnan(arr)] for arr in data_to_plot]

    plt.rc('font', size=9)
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False, widths=0.6)

    for i, group_name in enumerate(groups):
        color_key = group_name.lower()
        style = palette.get(color_key, {"face": "#cccccc", "edge": "#888888", "dot": "#666666"})

        bp["boxes"][i].set(facecolor=style["face"], edgecolor=style["edge"], linewidth=1.5)

        for part in ["whiskers", "caps"]:
            plt.setp(bp[part][i * 2:(i * 2) + 2], color="#6b6b6b", linewidth=1.1)
        plt.setp(bp["medians"][i], color="#222222", linewidth=2)

        rng = np.random.default_rng(42)
        y = data_to_plot[i]
        x = rng.normal(loc=i + 1, scale=0.08, size=len(y))
        ax.scatter(x, y, s=30, alpha=0.7, color=style["dot"],
                   edgecolors="black", linewidths=0.5, zorder=3)

    # Statistical comparison (Welch's t-test) for 2 groups
    if len(data_to_plot) == 2 and len(data_to_plot[0]) > 0 and len(data_to_plot[1]) > 0:
        stat_val, p = ttest_ind(data_to_plot[0], data_to_plot[1], equal_var=False)
        if p < 0.05:
            y_max = max([arr.max() for arr in data_to_plot if len(arr) > 0])
            y_min = min([arr.min() for arr in data_to_plot if len(arr) > 0])
            y_range = y_max - y_min if y_max != y_min else 1.0

            y_bracket = y_max + (y_range * 0.1)
            h = y_range * 0.05
            ax.plot([1, 1, 2, 2], [y_bracket, y_bracket + h, y_bracket + h, y_bracket], lw=1.2, c="black")

            stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*"
            ax.text(1.5, y_bracket + h, stars, ha="center", va="bottom", fontsize=12)
            ax.set_ylim(None, y_bracket + (y_range * 0.3))

    ax.set_title(title, fontsize=12, pad=15)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticklabels([g.upper() for g in groups])
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    return fig


def create_intensity_distribution(data: dict, title="Intensity Distribution", palette=None, figsize=(10, 5)):
    """
    Creates an intensity distribution plot with grouped bars for each intensity bin.

    Args:
        data: Dictionary with group names as keys and lists of images (numpy arrays) as values.
              Example: {"5x": [img1, img2, ...], "wt": [img1, img2, ...]}
        title: Plot title
        palette: Optional color palette dict. If None, uses DEFAULT_PALETTE.
        figsize: Tuple of (width, height) for figure size.

    Returns:
        matplotlib figure
    """
    palette = palette or DEFAULT_PALETTE
    n_bins = 16
    bin_edges = np.linspace(0, 256, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 256 / n_bins

    groups = list(data.keys())
    n_groups = len(groups)

    # Calculate histogram percentages for each image, then aggregate
    group_histograms = {}
    for group in groups:
        images = data[group]
        all_histograms = []

        for img in images:
            img_array = np.array(img).flatten()
            hist, _ = np.histogram(img_array, bins=bin_edges)
            hist_percent = (hist / len(img_array)) * 100
            all_histograms.append(hist_percent)

        if len(all_histograms) > 0:
            all_histograms = np.array(all_histograms)
            group_histograms[group] = {
                'median': np.median(all_histograms, axis=0),
                'q25': np.percentile(all_histograms, 25, axis=0),
                'q75': np.percentile(all_histograms, 75, axis=0)
            }

    plt.rc('font', size=9)
    fig, ax = plt.subplots(figsize=figsize)

    bar_width = bin_width / (n_groups + 1)
    offsets = np.linspace(-bar_width * (n_groups - 1) / 2, bar_width * (n_groups - 1) / 2, n_groups)

    for i, group in enumerate(groups):
        if group not in group_histograms:
            continue

        color_key = group.lower()
        style = palette.get(color_key, {"face": "#cccccc", "edge": "#888888", "dot": "#666666"})

        hist_data = group_histograms[group]
        x_positions = bin_centers + offsets[i]

        # Draw bars
        bars = ax.bar(
            x_positions,
            hist_data['median'],
            width=bar_width * 0.9,
            color=style['face'],
            edgecolor=style['edge'],
            linewidth=1,
            label=group.upper(),
            zorder=2
        )

        # Draw error lines (quartiles)
        for j, (x, median, q25, q75) in enumerate(zip(x_positions, hist_data['median'], hist_data['q25'], hist_data['q75'])):
            ax.plot([x, x], [q25, q75], color=style['edge'], linewidth=1.5, zorder=3)
            ax.plot([x - bar_width * 0.2, x + bar_width * 0.2], [q25, q25], color=style['edge'], linewidth=1.5, zorder=3)
            ax.plot([x - bar_width * 0.2, x + bar_width * 0.2], [q75, q75], color=style['edge'], linewidth=1.5, zorder=3)

    ax.set_xlabel("Pixel Intensity", fontsize=10)
    ax.set_ylabel("Percentage of Pixels (%)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=15)
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 50)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{int(e)}" for e in bin_edges[:-1]], rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc='upper right')
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    return fig

