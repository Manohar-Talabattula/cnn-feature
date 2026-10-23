
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions


# --------------------------------------------------
# 1. Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="CNN Image Feature Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CNN Image Feature Explorer")

st.write(
    "Upload an image to explore CNN feature maps "
    "and see the model's prediction."
)


# --------------------------------------------------
# 2. Load the pretrained CNN
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = MobileNetV2(
        weights="imagenet",
        include_top=True
    )
    return model


model = load_model()


# --------------------------------------------------
# 3. Create a feature extraction model
# --------------------------------------------------

layer_names = [
    "Conv1_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "out_relu"
]

feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=[
        model.get_layer(name).output
        for name in layer_names
    ]
)


# --------------------------------------------------
# 4. Upload an image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 5. Process the uploaded image
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")

    st.image(
        image,
        caption="Your uploaded image",
        width=350
    )

    # Resize the image to the model's expected size
    resized_image = image.resize((224, 224))

    # Convert the image into a NumPy array
    image_array = np.array(resized_image)

    # Add a batch dimension
    image_batch = np.expand_dims(image_array, axis=0)

    # Apply MobileNetV2 preprocessing
    processed_image = preprocess_input(
        image_batch.astype(np.float32)
    )


    # --------------------------------------------------
    # 6. Extract feature maps
    # --------------------------------------------------

    st.subheader("CNN Feature Maps")

    feature_maps = feature_model.predict(
        processed_image,
        verbose=0
    )

    for layer_name, feature_map in zip(
        layer_names,
        feature_maps
    ):

        st.markdown(f"### Layer: `{layer_name}`")

        st.write(
            f"Feature map shape: {feature_map.shape[1:]}"
        )

        # Display at most 8 channels from this layer
        number_of_channels = min(
            8,
            feature_map.shape[-1]
        )

        fig, axes = plt.subplots(
            2, 4,
            figsize=(12, 5)
        )

        for channel in range(8):

            ax = axes.flat[channel]
            ax.axis("off")

            if channel < number_of_channels:

                ax.set_title(f"Channel {channel}")

                ax.imshow(
                    feature_map[0, :, :, channel],
                    cmap="viridis"
                )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


    # --------------------------------------------------
    # 7. Predict the image class
    # --------------------------------------------------

    st.subheader("Image Classification")

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    results = decode_predictions(
        predictions,
        top=3
    )[0]

    for _, label, score in results:

        st.write(
            f"**{label.replace('_', ' ').title()}** "
            f"— {score * 100:.2f}%"
        )

        st.progress(float(score))


    # --------------------------------------------------
    # 8. Explain the results
    # --------------------------------------------------

    st.info(
        "Early layers often respond to edges and simple "
        "patterns. Deeper layers represent more complex "
        "visual features. The exact patterns depend on "
        "the trained model and the input image."
    )

else:

    st.info(
        "Upload an image to begin exploring the CNN."
    )
