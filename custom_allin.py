import time
import logging
import sys, os, random, re
from typing import Optional, List, Union, Tuple, Dict
from rich import print as rp
from rich.progress import track
from rich.console import Console
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QListWidget, QLineEdit, QPushButton, QTabWidget, QTextEdit, QFileDialog, QCheckBox,
    QSpinBox, QDoubleSpinBox, QDockWidget
)
from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont, QPixmap
from PyQt6.QtCore import QThread, Qt, pyqtSignal as Signal
from PureLLM import HugChatLLM

# Configure logging
logging.basicConfig(filename='all_in_one.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CodeHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for code in QTextEdit."""
    
    def __init__(self, document):
        super().__init__(document)
        self.keywords = {
            "def", "class", "import", "from", "if", "else", "elif",
            "for", "while", "try", "except", "return"
        }

    def highlightBlock(self, text):
        format = QTextCharFormat()
        format.setForeground(QColor("lightblue"))
        format.setFontWeight(QFont.Weight.Bold)

        for keyword in self.keywords:
            index = text.indexOf(keyword)
            while index >= 0:
                length = len(keyword)
                self.setFormat(index, length, format)
                index = text.indexOf(keyword, index + length)

class WaitThread(QThread):
    """Thread for waiting a specified amount of time."""

    finished = Signal()

    def __init__(self, wait_time: int):
        super().__init__()
        self.wait_time = wait_time

    def run(self):
        for _ in range(self.wait_time):
            time.sleep(1)
        self.finished.emit()

class ImageGenerator(HugChatLLM):
    """Class to handle image generation using the HugChatLLM."""

    def __init__(self, model_name: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct', client_models: list = ["black-forest-labs/FLUX.1-schnell","stabilityai/stable-diffusion-3-medium"]):
        super().__init__()
        self.flux_client = Client("black-forest-labs/FLUX.1-schnell")
        self.stability_client = Client("stabilityai/stable-diffusion-3-medium")
        self.llm = HugChatLLM().chatbot
        self.current_conv_id = None
        self.current_modelName = None
        self.current_modelIndex = 0
        self.hugchat_models = self.llm.get_available_llm_models()
        self.system_prompt_chatbot = '''
            You are an AI assistant. Here are your guidelines:
            - Respond to user input helpfully.
            - Use concise text, max 1024 tokens.
            - You are an expert in your field.

            Context for your reference:
            <<CONTEXT>>
            '''
        self.system_prompt_enhancer = '''
            Act as an 'image-prompt-enhancer':
            - Envision and describe AI-generated images vividly.
            - Provide detailed, first-person descriptions in one paragraph.
            - Use up to 40 tokens.

            Context:
            <<CONTEXT>>
            '''
        self._init_model(model_name)

    def _init_model(self, model_name):
        for model in self.hugchat_models:
            if model.name == model_name:
                self.current_modelIndex = self.hugchat_models.index(model)
                self.current_modelName = model_name
                break
            else:
                self.current_modelIndex = 0
                self.current_modelName = self.hugchat_models[self.current_modelIndex].name
                logging.warning(f"Model '{model_name}' not found. Defaulting to: {self.current_modelName}")

    def generate_image(self, prompt: str, steps: int = 5, size: int = 512, enhance_prompt_bool: bool = True, upscale: bool = True) -> Tuple[str, int]:
        path, seed = None, None
        while True:
            try:
                enhanced_prompt = self.enhance_prompt(prompt) if enhance_prompt_bool else prompt
                path, seed = self._try_generate_image(enhanced_prompt, steps, size)
                if upscale:
                    path = self._upscale_image(path, enhanced_prompt)
                break
            except AppError as e:
                if "GPU" in str(e):
                    return "GPU wait", self._handle_gpu_wait(str(e))
                else:
                    raise
        return path, seed

    def _try_generate_image(self, prompt, steps, size):
        return self.flux_client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=size,
            height=size,
            num_inference_steps=steps,
            api_name="/infer"
        )

    def _upscale_image(self, path, prompt):
        return Client("Manjushri/SD-2X-And-4X-CPU").predict(
            model="SD 2.0 2x Latent Upscaler",
            input_image=handle_file(path),
            prompt=prompt,
            guidance=0,
            api_name="/predict"
        )

    def _handle_gpu_wait(self, error_message):
        retry_time_match = re.search(r'retry in (\d+):(\d+):(\d+)', error_message)
        if retry_time_match:
            hours, minutes, seconds = map(int, retry_time_match.groups())
            wait_time = hours * 3600 + minutes * 60 + seconds + 1
            rp(f"GPU quota exceeded. Waiting {wait_time} seconds...")
            return wait_time
        return 30  # Default wait time


    def generate_new_images(self, prompt: str, enhance_prompt: bool = True) -> list[str]:
        """Generate new images using the new feature API."""
        # if we keep a folder of character images 'input_folder' can be set to let ai pick aux 1,2,3 if not locked
        input_folder = self.input_folder_line_edit.text()
        pos_prompt=self.generator.enhance_prompt(prompt) if enhance_prompt else prompt
        main_image_path = self.main_image_line_edit.text()
        aux_image_path_1 = self.aux_image_line_edit_1.text() if not self.aux_image_lock_1.isChecked() else None
        aux_image_path_2 = self.aux_image_line_edit_2.text() if not self.aux_image_lock_2.isChecked() else None
        aux_image_path_3 = self.aux_image_line_edit_3.text() if not self.aux_image_lock_3.isChecked() else None
        width = self.width_spin_box.value()
        height = self.height_spin_box.value()
        scale = self.scale_spin_box.value()
        steps = self.steps_spin_box.value()
        seed = self.seed_spin_box.value()
        num_samples = self.num_samples_spin_box.value()
        negative_prompt = self.negative_prompt_line_edit.text()

        try:
            # Call the new API to generate the images
            generated_images = self.call_new_api(
                
                main_image_path,
                aux_image_path_1,
                aux_image_path_2,
                aux_image_path_3,
                pos_prompt,
                negative_prompt,
                scale,
                steps,
                seed,
                num_samples,
                width,
                height,
                
            )
            # Update the chat history and artifacts
            self.chat_history.addItem("Bot: New images generated!")
            self.update_artifacts(generated_images, "Generated Images")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            self.chat_history.addItem(f"Bot: {error_message}")
            logging.error(error_message)

    def rindex(self, item_list): 
        return random.randint(0, len(item_list) - 1)
    
    def call_new_api(self, main_image_path: str, aux_image_path_1: str, aux_image_path_2: str, aux_image_path_3: str, prompt: str, negative_prompt: str , scale: float, steps: int, seed: int, num_samples: int, width: int, height: int, input_folder: str = None) -> List[Tuple[str, str]]:
        """Call the new API endpoint and return the generated images."""
        
        # Initialize the client with the provided API endpoint
        client = Client("https://yanze-pulid.hf.space/--replicas/zlhz9/")
        if input_folder and os.path.isdir(input_folder):
            image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
            
            aux_image_path_1 = image_files[self.rindex(image_files)] if aux_image_path_1 and not self.aux_image_lock_1.isChecked() else aux_image_path_1
            aux_image_path_2 = image_files[self.rindex(image_files)] if aux_image_path_2 and not self.aux_image_lock_1.isChecked() else aux_image_path_2
            aux_image_path_3 = image_files[self.rindex(image_files)] if aux_image_path_3 and not self.aux_image_lock_1.isChecked() else aux_image_path_3

        # Make the API call using the provided parameters
        result = client.predict(
            main_image_path,   # filepath in 'ID image (main)' Image component
            aux_image_path_1,  # filepath in 'Additional ID image (auxiliary)' Image component
            aux_image_path_2,  # filepath in 'Additional ID image (auxiliary)' Image component
            aux_image_path_3,  # filepath in 'Additional ID image (auxiliary)' Image component
            prompt,            # Replace with the actual prompt text if needed
            negative_prompt,   # Negative prompt for the API
            scale,             # CFG, recommend value range [1, 1.5]
            num_samples,       # Number of samples to generate
            seed,              # Seed for the generation
            steps,             # Number of steps for the generation
            height,            # Image height
            width,             # Image width
            0,                 # ID scale (use an appropriate value based on your requirement)
            "fidelity",        # Mode of operation (can be 'fidelity' or 'extremely style')
            True,              # ID Mix option (whether to mix two ID images)
            api_name="/run"    # API endpoint name
        )

        # Parse the result to extract the paths and captions of the generated images
        generated_images = []
        if result and isinstance(result[0], list):
            for item in result[0]:
                image_path = item.get("image", "")
                caption = item.get("caption", "Generated Image")
                generated_images.append((image_path, caption))

        # Return the list of tuples with image paths and captions
        return generated_images

    def update_artifacts(self, artifacts: Dict[str, List[Union[str, Tuple[str, str]]]]):
        """Update the artifact tabs with generated content."""
        self.tab_widget.clear()

        # Add the "New Feature" tab
        self.tab_widget.addTab(self.feature_tab, "New Feature")

        # Create the artifact group tabs
        for group_name, group_artifacts in artifacts.items():
            group_widget = QDockWidget(group_name)
            group_widget.setAllowedAreas(Qt.AllDockWidgetAreas)
            group_widget.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)

            group_contents = QWidget()
            group_layout = QVBoxLayout()
            group_contents.setLayout(group_layout)

            # Add the sub-tabs for each artifact type
            sub_tab_widget = QTabWidget()
            group_layout.addWidget(sub_tab_widget)

            # Image prompts tab
            image_prompts_tab = QWidget()
            image_prompts_layout = QVBoxLayout()
            image_prompts_tab.setLayout(image_prompts_layout)

            for image_prompt in [artifact for artifact in group_artifacts if isinstance(artifact, str)]:
                image_prompt_widget = QWidget()
                image_prompt_layout = QVBoxLayout()
                image_prompt_widget.setLayout(image_prompt_layout)

                text_edit = QTextEdit()
                text_edit.setPlainText(image_prompt)
                text_edit.setStyleSheet("background-color: black; color: white; font-family: Courier New; font-size: 12pt;")
                CodeHighlighter(text_edit.document())
                save_button = QPushButton("Save to File")
                save_button.clicked.connect(lambda checked, text_edit=text_edit: self.save_to_file(text_edit))

                image_prompt_layout.addWidget(text_edit)
                image_prompt_layout.addWidget(save_button)
                image_prompts_layout.addWidget(image_prompt_widget)

            sub_tab_widget.addTab(image_prompts_tab, "Image Prompts")

            # Images gallery tab
            images_gallery_tab = QWidget()
            images_gallery_layout = QVBoxLayout()
            images_gallery_tab.setLayout(images_gallery_layout)

            for image_path, image_title in [artifact for artifact in group_artifacts if isinstance(artifact, tuple)]:
                image_widget = QWidget()
                image_layout = QVBoxLayout()
                image_widget.setLayout(image_layout)

                scroll_area = QScrollArea()
                image_label = QLabel()
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                scroll_area.setWidget(image_label)
                scroll_area.setWidgetResizable(True)

                image_layout.addWidget(scroll_area)
                images_gallery_layout.addWidget(image_widget)

            sub_tab_widget.addTab(images_gallery_tab, "Images Gallery")

            # Code snippets tab
            code_snippets_tab = QWidget()
            code_snippets_layout = QVBoxLayout()
            code_snippets_tab.setLayout(code_snippets_layout)

            for code_snippet in [artifact for artifact in group_artifacts if isinstance(artifact, str)]:
                code_snippet_widget = QWidget()
                code_snippet_layout = QVBoxLayout()
                code_snippet_widget.setLayout(code_snippet_layout)

                text_edit = QTextEdit()
                text_edit.setPlainText(code_snippet)
                text_edit.setStyleSheet("background-color: black; color: white; font-family: Courier New; font-size: 12pt;")
                CodeHighlighter(text_edit.document())
                save_button = QPushButton("Save to File")
                save_button.clicked.connect(lambda checked, text_edit=text_edit: self.save_to_file(text_edit))

                code_snippet_layout.addWidget(text_edit)
                code_snippet_layout.addWidget(save_button)
                code_snippets_layout.addWidget(code_snippet_widget)

            sub_tab_widget.addTab(code_snippets_tab, "Code Snippets")

            group_widget.setWidget(group_contents)
            self.main_layout.addWidget(group_widget)

        # Set the current tab index to the "New Feature" tab
        self.tab_widget.setCurrentIndex(0)

    def update_artifacts_old(self, artifacts: List[Union[str, Tuple[str, str]]], image_path: Optional[str]):
        #Update the artifact tabs with generated content.
        self.tab_widget.clear()
        self.tab_widget.addTab(self.feature_tab, "New Feature")
        for i, artifact in enumerate(artifacts):
            tab = QWidget()
            layout = QVBoxLayout()

            if isinstance(artifact, str):
                # Text artifact
                text_edit = QTextEdit()
                text_edit.setPlainText(artifact)
                text_edit.setStyleSheet("background-color: black; color: white; font-family: Courier New; font-size: 12pt;")
                CodeHighlighter(text_edit.document())
                save_button = QPushButton("Save to File")
                save_button.clicked.connect(lambda checked, text_edit=text_edit: self.save_to_file(text_edit))
                layout.addWidget(text_edit)
                layout.addWidget(save_button)
                tab_title = f"Artifact {i+1}"
            elif isinstance(artifact, tuple):
                # Image artifact
                image_path, tab_title = artifact
                scroll_area = QScrollArea()
                image_label = QLabel()
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                scroll_area.setWidget(image_label)
                scroll_area.setWidgetResizable(True)
                layout.addWidget(scroll_area)

            tab.setLayout(layout)
            self.tab_widget.addTab(tab, tab_title) 

    def save_to_file(self, text_edit: QTextEdit):
        '''Save the content of a QTextEdit to a file.'''
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            with open(file_name, 'w') as file:
                file.write(text_edit.toPlainText())

if __name__ == "__main__":
    app = QApplication([])
    app.setStyleSheet("color: blue;"
                        "background-color: yellow;"
                        "selection-color: yellow;"
                        "selection-background-color: blue;"
                        )
    window = ChatApp()
    window.show()
    app.exec()

# improvements or additional features:
'''
Asynchronous Image Generation: The current implementation blocks the UI until the image generation is complete. Implementing asynchronous image generation, possibly using Qt's signal-slot mechanism, could provide a more responsive user experience.
Customizable Model Selection: Allow users to select the LLM model they want to use for image generation and normal chat processing, rather than relying on a predefined default.
Improved Error Handling: Expand the error handling to provide more user-friendly error messages and gracefully handle a wider range of exceptions.
Persistent Conversation History: Consider adding the ability to save and load the chat history, allowing users to resume their conversations.
Additional Artifact Types: Extend the update_artifacts method to handle more types of artifacts, such as Mermaid diagrams or React components, to provide a richer set of tools for the users.
Performance Optimization: Investigate ways to optimize the image generation process, such as caching previous results or leveraging GPU acceleration more efficiently.
'''
