import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3

# ====== Load pretrained model ======
base_model = inception_v3.InceptionV3(weights="imagenet", include_top=False)
dream_layers = ['mixed3', 'mixed5', 'mixed7']
outputs = [base_model.get_layer(name).output for name in dream_layers]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

# ====== Helper functions ======
def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if not isinstance(layer_activations, list):
        layer_activations = [layer_activations]
    losses = [tf.reduce_mean(act[:, 2:-2, 2:-2, :]) for act in layer_activations]
    return tf.reduce_sum(losses)

@tf.function
def gradient_ascent_step(img, model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    gradients = tape.gradient(loss, img)
    gradients /= (tf.math.reduce_std(gradients) + 1e-8)
    img = img + gradients * step_size
    img = tf.clip_by_value(img, -1, 1)
    return img, loss

def run_deep_dream(img, model, steps=100, step_size=0.01):
    img = tf.convert_to_tensor(img)
    for _ in range(steps):
        img, _ = gradient_ascent_step(img, model, step_size)
    return img

def resize_img(img, size):
    return tf.image.resize(img, size)

def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)

def run_deep_dream_octaves(img, model,
                           step_size=0.01, steps_per_octave=100,
                           octaves=3, octave_scale=1.4):
    base_shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    float_img = tf.convert_to_tensor(img)
    for _ in range(octaves):
        new_size = tf.cast(base_shape * (octave_scale ** _), tf.int32)
        float_img = resize_img(float_img, new_size)
        float_img = run_deep_dream(float_img, model,
                                   steps=steps_per_octave,
                                   step_size=step_size)
    return float_img

# ====== Main Dreamify function ======
def dreamify(uploaded_image, steps, step_size, octaves, octave_scale):
    img = uploaded_image.resize((299, 299))
    img_np = np.array(img)
    img_pre = inception_v3.preprocess_input(img_np)
    dreamed = run_deep_dream_octaves(
        img_pre, dream_model,
        steps_per_octave=int(steps),
        step_size=float(step_size),
        octaves=int(octaves),
        octave_scale=float(octave_scale),
    )
    dreamed_out = deprocess(dreamed).numpy()
    dreamed_img = Image.fromarray(dreamed_out)
    return img, dreamed_img

# ====== Gradio UI ======
with gr.Blocks(title="🌀 DeepDream Visualizer") as demo:
    gr.Markdown("## 🧠 DeepDream Visualizer")
    gr.Markdown("Upload an image and watch how your CNN *dreams* on it. Adjust parameters for intensity and scale.")

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload an image")
        with gr.Column():
            steps = gr.Slider(20, 200, 100, step=10, label="Steps per octave")
            step_size = gr.Slider(0.002, 0.02, 0.01, step=0.002, label="Step size (dream intensity)")
            octaves = gr.Slider(1, 5, 3, step=1, label="Number of octaves")
            octave_scale = gr.Slider(1.2, 1.6, 1.4, step=0.05, label="Octave scale (zoom per level)")
            run_btn = gr.Button("✨ Dreamify!")

    with gr.Row():
        out_original = gr.Image(label="Original Image")
        out_dream = gr.Image(label="Dreamified Output", show_download_button=True)

    run_btn.click(fn=dreamify,
                  inputs=[inp, steps, step_size, octaves, octave_scale],
                  outputs=[out_original, out_dream])

if __name__ == "__main__":
    demo.launch()
