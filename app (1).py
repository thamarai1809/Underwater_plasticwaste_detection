import gradio as gr
from utils import detect_objects
from PIL import Image
import numpy as np
import cv2

def detect_and_annotate(image):
    temp_path = "temp.jpg"
    image.save(temp_path)
    boxes, class_ids, confidences, class_names = detect_objects(temp_path)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Custom CSS styling
custom_css = """
.gradio-container { 
    background-image: url('https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/underwater-view-of-pebbles-lake-baikal-siberia-russia-michel-roggo--natureplcom.jpg'); 
    background-size: cover; 
    background-position: center; 
    font-family: Arial, sans-serif;
}

h1 {
    text-align: center;
    font-weight: bold;
    font-size: 36px;
    color: white;
    margin-bottom: 10px;
}

.gr-image-box, .gr-box {
    background-color: rgba(0, 0, 0, 0.6); /* Dark semi-transparent */
    border-radius: 10px;
    padding: 15px;
    border: 2px solid #00b894;
}

.gr-label {
    font-size: 18px;
    font-weight: bold;
    color: white;
}

.gr-button {
    background-color: #00b894;
    color: white;
    font-weight: bold;
}

.gr-button:hover {
    background-color: #009e7f;
}

#share-btn {
    background-color: #4CAF50; 
    color: white;
    padding: 10px 16px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 16px;
    margin-top: 10px;
}

#share-btn:hover {
    background-color: #45a049;
}

#copy-status {
    color: white;
    text-align: center;
    margin-top: 5px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    # 🌊 **Underwater Plastic Waste Detection**
    Upload an underwater image to detect plastic.   
    """)

    with gr.Row():
        input_image = gr.Image(type="pil", label="🖼️ Upload Image")
        output_image = gr.Image(type="numpy", label="🎯 Detection Result")

    detect_btn = gr.Button("🔍 Run Detection")
    detect_btn.click(fn=detect_and_annotate, inputs=input_image, outputs=output_image)

    gr.HTML("""
    <div style='text-align:center;'>
        <button id="share-btn" onclick="copyURL()">🔗 Share This App</button>
        <p id="copy-status"></p>
    </div>
    <script>
    function copyURL() {
        navigator.clipboard.writeText(window.location.href).then(function() {
            document.getElementById("copy-status").innerText = "✅ Link copied to clipboard!";
        }, function(err) {
            document.getElementById("copy-status").innerText = "❌ Failed to copy link.";
        });
    }
    </script>
    """)

if __name__ == "__main__":
    demo.launch()
