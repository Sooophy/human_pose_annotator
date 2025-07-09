import sys
import json
import os
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QListWidget, QGraphicsView, QGraphicsScene, QSlider,
                           QSpinBox, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF

class VideoProcessor:
    def __init__(self):
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        
    def load_video(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def get_frame(self, frame_number):
        if self.cap is None:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def save_frame(self, frame_number, output_dir, image_id):
        frame = self.get_frame(frame_number)
        if frame is not None:
            # Format filename with 12 digits using image_id (COCO format)
            filename = f"{image_id:012d}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return filename
        return None
    
    def close(self):
        if self.cap is not None:
            self.cap.release()

class KeypointScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keypoints = {}  # Dictionary to store keypoints {name: (x, y, v)}
        self.current_keypoint = None
        self.keypoint_items = {}  # Store visual items for each keypoint
        self.keypoint_updated = None  # Callback for keypoint updates
        self.bbox_item = None  # Store the bounding box visual item
        
    def set_current_keypoint(self, keypoint_name):
        self.current_keypoint = keypoint_name
        # Highlight the currently selected keypoint
        self.update_keypoint_visuals()
        
    def mousePressEvent(self, event):
        if self.current_keypoint:
            pos = event.scenePos()
            self.keypoints[self.current_keypoint] = (pos.x(), pos.y(), 2)
            self.update_keypoint_visuals()
            
            # Update keypoint status in the list
            if hasattr(self, 'keypoint_updated'):
                self.keypoint_updated(self.current_keypoint, True)
                
            # Update bounding box
            self.update_bounding_box()
    
    def calculate_bbox(self):
        """Calculate bounding box from keypoints"""
        if not self.keypoints:
            return None
            
        valid_x = [x for x, y, v in self.keypoints.values()]
        valid_y = [y for x, y, v in self.keypoints.values()]
        
        if valid_x and valid_y:
            x_min, x_max = min(valid_x), max(valid_x)
            y_min, y_max = min(valid_y), max(valid_y)
            
            # Add padding to make box slightly larger than the keypoints
            padding = 30
            x_min -= padding
            y_min -= padding
            x_max += padding
            y_max += padding
            
            return [x_min, y_min, x_max - x_min, y_max - y_min]
        return None
    
    def update_bounding_box(self):
        """Update the bounding box visualization"""
        # Remove existing bbox if any
        if self.bbox_item:
            self.removeItem(self.bbox_item)
            self.bbox_item = None
        
        bbox = self.calculate_bbox()
        if bbox:
            # Draw new bbox with dashed green line
            pen = QPen(QColor(0, 255, 0))  # Green color
            pen.setStyle(Qt.DashLine)  # Dashed line
            pen.setWidth(2)  # Line width
            
            self.bbox_item = self.addRect(bbox[0], bbox[1], bbox[2], bbox[3], pen)
    
    def update_keypoint_visuals(self):
        # Clear all existing keypoint visualizations
        for items in self.keypoint_items.values():
            for item in items:
                self.removeItem(item)
        self.keypoint_items.clear()
        
        # Redraw all keypoints
        for kp_name, (x, y, v) in self.keypoints.items():
            items = []
            # Current keypoint gets highlighted in red
            color = QColor(255, 0, 0) if kp_name == self.current_keypoint else QColor(0, 255, 0)
            
            ellipse = self.addEllipse(x-3, y-3, 6, 6, QPen(color), color)
            text = self.addText(kp_name)
            text.setPos(x+5, y+5)
            
            items.extend([ellipse, text])
            self.keypoint_items[kp_name] = items
        
        # Update bounding box
        self.update_bounding_box()
    
    def reset_keypoint(self, keypoint_name):
        if keypoint_name in self.keypoints:
            del self.keypoints[keypoint_name]
            self.update_keypoint_visuals()
            if hasattr(self, 'keypoint_updated'):
                self.keypoint_updated(keypoint_name, False)

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(KeypointScene())
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

class PoseLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_processor = VideoProcessor()
        self.current_frame_number = 0
        self.output_dir = None
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        self.initUI()
        self.annotations = {
            "info": {
                "description": "Pose Keypoint Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{
                "url": "",
                "id": 1,
                "name": ""
            }],
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": self.keypoint_names,
                "skeleton": [
                    [16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13], 
                    [6,7], [6,8], [7,9], [8,10], [9,11], [2,3], [1,2], [1,3], 
                    [2,4], [3,5], [4,6], [5,7]
                ]
            }]
        }
        
    def initUI(self):
        self.setWindowTitle('Video Frame Pose Labeling Tool')
        self.setGeometry(100, 100, 1400, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create image viewer
        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, stretch=2)
        
        # Create right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        layout.addWidget(right_panel, stretch=1)
        
        # Video controls
        load_video_btn = QPushButton('Load Video')
        load_video_btn.clicked.connect(self.loadVideo)
        right_layout.addWidget(load_video_btn)
        
        # Frame navigation
        frame_control = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.updateFrame)
        frame_control.addWidget(self.frame_slider)
        
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.valueChanged.connect(self.updateFrame)
        frame_control.addWidget(self.frame_spinbox)
        right_layout.addLayout(frame_control)
        
        # Keypoint list
        right_layout.addWidget(QLabel('Keypoints:'))
        self.keypoint_list = QListWidget()
        self.keypoint_list.addItems(self.keypoint_names)
        self.keypoint_list.setFixedHeight(
            self.keypoint_list.sizeHintForRow(0) * len(self.keypoint_names) + 10)
        self.keypoint_list.currentTextChanged.connect(
            lambda x: self.viewer.scene().set_current_keypoint(x))
        self.keypoint_list.setCurrentRow(0)
        right_layout.addWidget(self.keypoint_list)
        
        # Reset keypoint button
        reset_keypoint_btn = QPushButton('Reset Selected Keypoint')
        reset_keypoint_btn.clicked.connect(self.resetSelectedKeypoint)
        right_layout.addWidget(reset_keypoint_btn)
        
        # Save controls
        set_output_btn = QPushButton('Set Output Directory')
        set_output_btn.clicked.connect(self.setOutputDirectory)
        right_layout.addWidget(set_output_btn)
        
        save_btn = QPushButton('Save Current Frame')
        save_btn.clicked.connect(self.saveAnnotations)
        right_layout.addWidget(save_btn)
        
        reset_btn = QPushButton('Reset All Keypoints')
        reset_btn.clicked.connect(self.resetCurrent)
        right_layout.addWidget(reset_btn)
        
        # Method to update keypoint status
        def update_keypoint_status(keypoint_name, is_labeled):
            items = self.keypoint_list.findItems(keypoint_name, Qt.MatchExactly)
            if items:
                item = items[0]
                if is_labeled:
                    item.setBackground(QColor(200, 255, 200))
                else:
                    item.setBackground(QColor(255, 255, 255))
                    
        self.viewer.scene().keypoint_updated = update_keypoint_status
            
    def loadVideo(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_processor.load_video(video_path)
            self.frame_slider.setMaximum(self.video_processor.total_frames - 1)
            self.frame_spinbox.setMaximum(self.video_processor.total_frames - 1)
            self.updateFrame(0)
            
    def updateFrame(self, frame_number):
        self.current_frame_number = frame_number
        self.frame_slider.setValue(frame_number)
        self.frame_spinbox.setValue(frame_number)
        
        frame = self.video_processor.get_frame(frame_number)
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Create new scene
            new_scene = KeypointScene()
            self.viewer.setScene(new_scene)
            
            # Reset all keypoint highlights to white
            for i in range(self.keypoint_list.count()):
                self.keypoint_list.item(i).setBackground(QColor(255, 255, 255))
            
            # Set up keypoint update callback
            def update_keypoint_status(keypoint_name, is_labeled):
                items = self.keypoint_list.findItems(keypoint_name, Qt.MatchExactly)
                if items:
                    item = items[0]
                    if is_labeled:
                        item.setBackground(QColor(200, 255, 200))
                    else:
                        item.setBackground(QColor(255, 255, 255))
            
            new_scene.keypoint_updated = update_keypoint_status
            
            # Add image to scene
            pixmap = QPixmap.fromImage(q_image)
            new_scene.addPixmap(pixmap)
            from PyQt5.QtCore import QRectF
            self.viewer.setSceneRect(QRectF(pixmap.rect()))
            self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)
            
            # Preserve the selected keypoint
            current_keypoint = self.keypoint_list.currentItem().text()
            new_scene.set_current_keypoint(current_keypoint)
                
    def resetSelectedKeypoint(self):
        current_keypoint = self.keypoint_list.currentItem().text()
        self.viewer.scene().reset_keypoint(current_keypoint)
        
    def setOutputDirectory(self):
        self.output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if self.output_dir:
            os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        
    def saveAnnotations(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please set output directory first!")
            return
            
        # Check if this frame from this video has already been annotated
        existing_annotation = None
        existing_image = None
        for i, img in enumerate(self.annotations["images"]):
            if (img["video_path"] == self.video_processor.video_path and 
                img["frame_number"] == self.current_frame_number):
                existing_image = img
                existing_annotation = self.annotations["annotations"][i]
                break
                
        if existing_annotation:
            reply = QMessageBox.question(self, 'Duplicate Frame',
                                       f'Frame {self.current_frame_number} from this video has already been annotated. '
                                       'Do you want to replace the existing annotation?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # Remove existing annotation and image
                self.annotations["images"].remove(existing_image)
                self.annotations["annotations"].remove(existing_annotation)
            else:
                QMessageBox.information(self, "Skipped", "Frame annotation skipped.")
                return
            
        frames_dir = os.path.join(self.output_dir, "frames")
        # Calculate next image ID
        image_id = len(self.annotations["images"]) + 1
        filename = self.video_processor.save_frame(self.current_frame_number, frames_dir, image_id)
        
        if filename:
            image_info = {
                "id": image_id,
                "file_name": filename,
                "video_path": self.video_processor.video_path,
                "frame_number": self.current_frame_number,
                "width": self.video_processor.frame_width,
                "height": self.video_processor.frame_height,
                "fps": self.video_processor.fps,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            keypoints = []
            for kp_name in self.keypoint_names:
                if kp_name in self.viewer.scene().keypoints:
                    x, y, v = self.viewer.scene().keypoints[kp_name]
                    keypoints.extend([x, y, v])
                else:
                    keypoints.extend([0, 0, 0])
                
            # Calculate bbox from keypoints
            valid_x = [x for x, y, v in self.viewer.scene().keypoints.values()]
            valid_y = [y for x, y, v in self.viewer.scene().keypoints.values()]
            if valid_x and valid_y:  # If there are any keypoints
                x_min, x_max = min(valid_x), max(valid_x)
                y_min, y_max = min(valid_y), max(valid_y)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                bbox = [x_min, y_min, bbox_width, bbox_height]
                area = bbox_width * bbox_height
            else:
                bbox = [0, 0, 0, 0]
                area = 0

            annotation = {
                "id": len(self.annotations["annotations"]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": len(self.viewer.scene().keypoints),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],  # Empty for keypoint-only annotations
                "score": 1.0  # Detection confidence score
            }
            
            self.annotations["images"].append(image_info)
            self.annotations["annotations"].append(annotation)
            
            with open(os.path.join(self.output_dir, 'annotations.json'), 'w') as f:
                json.dump(self.annotations, f, indent=2)
            
            QMessageBox.information(self, "Success", 
                                  f"Frame {self.current_frame_number} saved successfully!")
                
    def resetCurrent(self):
        self.updateFrame(self.current_frame_number)
        
    def closeEvent(self, event):
        self.video_processor.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    labeler = PoseLabeler()
    labeler.show()
    sys.exit(app.exec_())