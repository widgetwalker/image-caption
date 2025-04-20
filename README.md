AI-Powered Image Caption Generator
AI-powered Image Caption Generator using the BLIP model. Upload images or capture via webcam to generate descriptive captions. Captions and images are stored in a SQLite database with a history view. Built with Streamlit, PyTorch, and OpenCV.
Features

Generate captions using Salesforce BLIP model
Supports image uploads and webcam capture
Stores captions in SQLite database
Displays caption history with images
Streamlit-based web interface

Prerequisites

Python 3.8+
Webcam (optional, for capture feature)
Git

Installation

Clone the repository:git clone https://github.com/username/ai-image-caption-generator.git


Navigate to the project directory:cd ai-image-caption-generator


Install dependencies:pip install -r requirements.txt


Run the application:streamlit run main.py



Usage

Web Interface: Open http://localhost:8501 * Upload images (JPG, JPEG, PNG) or capture from webcam to generate captions


View caption history in the "Caption History" section


Database: Captions and image paths are stored in caption_database.db
Images: Saved in the images/ folder

Project Structure

main.py: Main application script
requirements.txt: Project dependencies
.gitignore: Excludes caption_database.db, images/, and Python cache files
LICENSE: MIT License

Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Salesforce BLIP for image captioning
Streamlit for the web interface
PyTorch and Transformers for model inference

