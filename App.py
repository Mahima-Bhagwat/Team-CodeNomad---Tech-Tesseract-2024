import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QVBoxLayout, QWidget, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap
from  ultralytics import YOLO 

class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Scientific Image Classifier")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        self.image_path_label = QLabel("Image Path:")
        self.image_path_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.predict_button = QPushButton("Predict")
        self.result_label = QLabel()
        self.image_label = QLabel()

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_path_label)
        vbox.addWidget(self.image_path_input)
        vbox.addWidget(self.browse_button)
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.predict_button)
        vbox.addWidget(self.result_label)

        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        self.browse_button.clicked.connect(self.browse_image)
        self.predict_button.clicked.connect(self.predict_image)

    def browse_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            self.image_path_input.setText(filename)
            pixmap = QPixmap(filename)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def predict_image(self):
        image_path = self.image_path_input.text()
        if not image_path:
            QMessageBox.warning(self, "Warning", "Please provide the path to an image.")
            return

        try:
            model = YOLO('D:/Hackathons/MINeD/train2/weights/best.pt')  
            results = model(image_path)  # Predict on the image
            names_dict = results[0].names
            probs = results[0].probs.data.tolist()
            predicted_class = names_dict[np.argmax(probs)]
            self.result_label.setText(f"Predicted class: {predicted_class}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
