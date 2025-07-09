import sys
import json
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QComboBox, QGraphicsView, QGraphicsScene, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF

class AnnotationViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene())
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setFrameShape(QGraphicsView.NoFrame)
        
    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

class AnnotationVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.annotations_data = None
        self.frames_dir = None
        # COCO Keypoint Colors
        self.keypoint_colors = {
            "nose": QColor(255, 0, 0),      # Red
            "left_eye": QColor(255, 85, 0),  
            "right_eye": QColor(255, 170, 0),
            "left_ear": QColor(255, 255, 0),  
            "right_ear": QColor(170, 255, 0),
            "left_shoulder": QColor(85, 255, 0),
            "right_shoulder": QColor(0, 255, 0),
            "left_elbow": QColor(0, 255, 85),   
            "right_elbow": QColor(0, 255, 170),
            "left_wrist": QColor(0, 255, 255),  
            "right_wrist": QColor(0, 170, 255),
            "left_hip": QColor(0, 85, 255),    
            "right_hip": QColor(0, 0, 255),    
            "left_knee": QColor(85, 0, 255),   
            "right_knee": QColor(170, 0, 255),
            "left_ankle": QColor(255, 0, 255),
            "right_ankle": QColor(255, 0, 170)
        }
        self.skeleton_color = QColor(0, 128, 255)  # Light blue
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Pose Annotation Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create image viewer
        self.viewer = AnnotationViewer()
        layout.addWidget(self.viewer, stretch=2)
        
        # Create right panel for controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        layout.addWidget(right_panel, stretch=1)
        
        # Add select frames directory button first
        frames_btn = QPushButton('Select Frames Directory')
        frames_btn.clicked.connect(self.selectFramesDirectory)
        right_layout.addWidget(frames_btn)
        
        # Then add load annotations button
        load_btn = QPushButton('Load Annotations')
        load_btn.clicked.connect(self.loadAnnotations)
        right_layout.addWidget(load_btn)
        
        # Add image selector combo box
        right_layout.addWidget(QLabel('Select Image:'))
        self.image_selector = QComboBox()
        self.image_selector.currentIndexChanged.connect(self.displaySelectedAnnotation)
        right_layout.addWidget(self.image_selector)
        
        # Add info display
        self.info_label = QLabel()
        right_layout.addWidget(self.info_label)
        
        # Add spacer at the bottom
        right_layout.addStretch()
        
    def loadAnnotations(self):
        annotations_file, _ = QFileDialog.getOpenFileName(
            self, "Select Annotations File", "", "JSON Files (*.json)")
        
        if not annotations_file:
            return
            
        with open(annotations_file, 'r') as f:
            self.annotations_data = json.load(f)
            
        # Try to find frames in default location first
        default_frames_dir = os.path.join(os.path.dirname(annotations_file), "frames")
        if os.path.exists(default_frames_dir):
            self.frames_dir = default_frames_dir
        else:
            reply = QMessageBox.question(self, 'Frames Directory Not Found',
                                       'Cannot find "frames" folder in the default location.\n'
                                       'Would you like to select the frames directory manually?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.Yes:
                self.selectFramesDirectory()
                
        self.image_selector.clear()
        for image in self.annotations_data['images']:
            self.image_selector.addItem(f"ID: {image['id']} - Frame: {image['frame_number']}")
            
    def selectFramesDirectory(self):
        frames_dir = QFileDialog.getExistingDirectory(
            self, "Select Frames Directory")
        if frames_dir:
            self.frames_dir = frames_dir
            # If we already have an image selected, refresh the display
            if self.image_selector.currentIndex() >= 0:
                self.displaySelectedAnnotation(self.image_selector.currentIndex())
                
    def displaySelectedAnnotation(self, index):
        if index < 0 or not self.annotations_data:
            return
            
        if not self.frames_dir:
            QMessageBox.warning(self, "Error", "Please select frames directory first!")
            self.selectFramesDirectory()
            return
            
        # Get image and annotation data
        image_data = self.annotations_data['images'][index]
        annotation_data = next(
            (ann for ann in self.annotations_data['annotations'] 
             if ann['image_id'] == image_data['id']), None)
        
        if not annotation_data:
            return
            
        # Load and display image
        image_path = os.path.join(self.frames_dir, image_data['file_name'])
        frame = cv2.imread(image_path)
        if frame is None:
            return
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Clear and set up scene
        self.viewer.scene().clear()
        pixmap = QPixmap.fromImage(q_image)
        self.viewer.scene().addPixmap(pixmap)
        rect = pixmap.rect()
        self.viewer.setSceneRect(QRectF(rect.x(), rect.y(), rect.width(), rect.height()))
        self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)
        
        # Draw bounding box first
        bbox = annotation_data.get('bbox', None)
        if bbox and any(bbox):  # Check if bbox exists and is not all zeros
            pen = QPen(QColor(255, 165, 0))  # Orange color for bbox
            pen.setStyle(Qt.DashLine)  # Dashed line
            pen.setWidth(2)
            x, y, w, h = bbox
            self.viewer.scene().addRect(x, y, w, h, pen)
        
        # Get keypoint names and skeleton
        category = next((cat for cat in self.annotations_data['categories'] 
                        if cat['id'] == annotation_data['category_id']), None)
        if not category:
            return
            
        keypoint_names = category['keypoints']
        skeleton = category['skeleton']
        
        # Draw skeleton connections
        keypoints = annotation_data['keypoints']
        for connection in skeleton:
            start_idx = connection[0] - 1
            end_idx = connection[1] - 1
            
            start_x = keypoints[start_idx * 3]
            start_y = keypoints[start_idx * 3 + 1]
            start_v = keypoints[start_idx * 3 + 2]
            
            end_x = keypoints[end_idx * 3]
            end_y = keypoints[end_idx * 3 + 1]
            end_v = keypoints[end_idx * 3 + 2]
            
            if start_v > 0 and end_v > 0:
                pen = QPen(self.skeleton_color)
                pen.setWidth(2)
                self.viewer.scene().addLine(start_x, start_y, end_x, end_y, pen)
        
        # Draw keypoints
        visible_keypoints = 0
        labeled_but_invisible = 0
        for i, kp_name in enumerate(keypoint_names):
            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            v = keypoints[i * 3 + 2]
            
            if v > 0:
                base_color = self.keypoint_colors[kp_name]
                
                if v == 2:  # Visible
                    color = base_color
                    visible_keypoints += 1
                else:  # v == 1, Labeled but not visible
                    color = QColor(base_color.red(), base_color.green(), 
                                 base_color.blue(), 128)
                    labeled_but_invisible += 1
                
                # Draw point
                pen = QPen(color)
                pen.setWidth(2)
                self.viewer.scene().addEllipse(x-3, y-3, 6, 6, pen, color)
                
                # Add label
                text = self.viewer.scene().addText(kp_name)
                text.setDefaultTextColor(color)
                text.setPos(x+5, y+5)
        
        # Update info display
        bbox_info = f"BBox: x={bbox[0]:.1f}, y={bbox[1]:.1f}, w={bbox[2]:.1f}, h={bbox[3]:.1f}\n" if bbox else "No bbox\n"
        self.info_label.setText(
            f"Frame: {image_data['frame_number']}\n"
            f"Image ID: {image_data['id']}\n"
            f"{bbox_info}"
            f"Visible Keypoints: {visible_keypoints}\n"
            f"Labeled but Invisible: {labeled_but_invisible}\n"
            f"Total Keypoints: {len(keypoint_names)}"
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    visualizer = AnnotationVisualizer()
    visualizer.show()
    sys.exit(app.exec_())