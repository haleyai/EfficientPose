import numpy as np
from flask import Flask, jsonify, request

import track

app = Flask(__name__)


def inference_on_image(image, model_name="i_lite", framework_name="tflite"):
    """
    Run EfficientPose inference on an image.

    Args:
        image: path or array like
            System path of image to analyze or image array
        model_name: deep learning model
            EfficientPose model to utilize (rt, i, ii, iii, iv, rt_lite, i_lite or ii_lite)
        framework_name: string
            Deep learning framework to use (keras, tensorflow, tensorflow lite or pytorch)

    Returns:
        Predicted pose coordinates in the supplied image.

    """
    assert model_name in [
        "efficientposert",
        "rt",
        "efficientposei",
        "i",
        "efficientposeii",
        "ii",
        "efficientposeiii",
        "iii",
        "efficientposeiv",
        "iv",
        "efficientposert_lite",
        "rt_lite",
        "efficientposei_lite",
        "i_lite",
        "efficientposeii_lite",
        "ii_lite",
    ]
    assert framework_name in [
        "keras",
        "k",
        "tensorflow",
        "tf",
        "tensorflowlite",
        "tflite",
        "pytorch",
        "torch",
    ]

    model_name = model_name[13:] if len(model_name) > 7 else model_name
    lite = True if model_name.endswith("_lite") else False
    model, resolution = track.get_model(framework_name, model_name)
    assert model
    return track.analyze_image(image, model, framework_name, resolution, lite)


def bounding_box(coords_xy):
    xy_top_left = coords_xy.min(axis=0)
    xy_bot_right = coords_xy.max(axis=0)
    return np.concatenate((xy_top_left, xy_bot_right))


@app.post("/")
def process_image():
    coordinates = inference_on_image(request.files["file"])
    coordinates = coordinates[0]  # single image, not a sequence
    bbox = bounding_box(np.array([joint[1:] for joint in coordinates]))
    return jsonify({"joints": coordinates, "bounding_box_x1y1x2y2": tuple(bbox)})


if __name__ == "__main__":
    from PIL import Image

    img = np.array(Image.open("utils/MPII.jpg"))
    coordinates = inference_on_image(img)
