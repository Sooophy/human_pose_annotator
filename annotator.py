import sys
import json
import os
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QListWidget, QGraphicsView, QGraphicsScene, QSlider,
                           QSpinBox, QMessageBox, QComboBox, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QTextCursor
from PyQt5.QtCore import Qt, QPointF, QRectF

from pose_config import*


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
            # return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
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

class ImageViewer(QGraphicsView):
    def __init__(self, pose_config, parent=None):
        super().__init__(parent)
        self.setScene(KeypointScene(pose_config))
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
            
class KeypointScene(QGraphicsScene):
    def __init__(self, pose_config, parent=None):
        super().__init__(parent)
        self.pose_config = pose_config
        self.keypoints = {}
        self.current_keypoint = None
        self.keypoint_items = {}
        self.keypoint_updated = None
        self.bbox_item = None
        self.skeleton_lines = []  # Add this line to track skeleton lines
        self.editing_enabled = True
        
        # Colors for different states
        self.highlighted_color = QColor(255, 255, 0)  # Yellow for highlighted
        self.visible_color = QColor(0, 255, 0)       # Green for visible
        self.invisible_color = QColor(255, 165, 0)    # Orange for invisible but labeled

    def mousePressEvent(self, event):
        if not self.editing_enabled:
            return
            
        if self.current_keypoint:
            pos = event.scenePos()
            # Right click for visibility=1 (labeled but not visible)
            if event.button() == Qt.RightButton:
                self.keypoints[self.current_keypoint] = (pos.x(), pos.y(), 1)
            # Left click for visibility=2 (visible)
            elif event.button() == Qt.LeftButton:
                self.keypoints[self.current_keypoint] = (pos.x(), pos.y(), 2)
                
            self.update_keypoint_visuals()
            
            if self.keypoint_updated:
                self.keypoint_updated(self.current_keypoint, True)
            
            self.update_bounding_box()

    def update_keypoint_visuals(self):
        # Clear existing visualizations
        for items in self.keypoint_items.values():
            for item in items:
                self.removeItem(item)
        self.keypoint_items.clear()
        
        # Draw skeleton first
        self.draw_skeleton()
        
        # Draw keypoints
        for kp_name, (x, y, v) in self.keypoints.items():
            items = []
            
            # Use colors from pose_config instead of self.keypoint_colors
            base_color = self.pose_config.keypoint_colors.get(kp_name, QColor(0, 255, 0))
            if kp_name == self.current_keypoint:
                color = QColor(255, 255, 0)  # Highlight in yellow
            else:
                color = base_color
            
            # Adjust opacity based on visibility
            if v == 1:  # Labeled but not visible
                color.setAlpha(128)
            
            # Draw point
            ellipse = self.addEllipse(x-3, y-3, 6, 6, QPen(color), color)
            text = self.addText(kp_name)
            text.setDefaultTextColor(color)
            text.setPos(x+5, y+5)
            
            items.extend([ellipse, text])
            self.keypoint_items[kp_name] = items
        
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

    def set_current_keypoint(self, keypoint_name):
        """Set the currently selected keypoint and update visuals"""
        self.current_keypoint = keypoint_name
        # Highlight the currently selected keypoint
        self.update_keypoint_visuals()

    def reset_keypoint(self, keypoint_name):
        """Reset (remove) a specific keypoint"""
        if keypoint_name in self.keypoints:
            del self.keypoints[keypoint_name]
            self.update_keypoint_visuals()
            if hasattr(self, 'keypoint_updated'):
                self.keypoint_updated(keypoint_name, False)    
    
    
    # Add skeleton drawing functionality
    def draw_skeleton(self):
        # Clear existing skeleton lines
        for line in self.skeleton_lines:
            self.removeItem(line)
        self.skeleton_lines.clear()
        
        keypoints_list = []
        for kp_name in self.pose_config.keypoint_names:
            if kp_name in self.keypoints:
                x, y, v = self.keypoints[kp_name]
                keypoints_list.append((x, y, v))
            else:
                keypoints_list.append((0, 0, 0))
        
        for connection in self.pose_config.skeleton:
            start_idx = connection[0] - 1
            end_idx = connection[1] - 1
            
            if (start_idx < len(keypoints_list) and end_idx < len(keypoints_list)):
                start_x, start_y, start_v = keypoints_list[start_idx]
                end_x, end_y, end_v = keypoints_list[end_idx]
                
                if start_v > 0 and end_v > 0:
                    pen = QPen(self.pose_config.skeleton_color)
                    pen.setWidth(2)
                    line = self.addLine(start_x, start_y, end_x, end_y, pen)
                    self.skeleton_lines.append(line)  # Store the line

    
    def update_bounding_box(self):
        if self.bbox_item:
            self.removeItem(self.bbox_item)
            self.bbox_item = None
        
        bbox = self.calculate_bbox()
        if bbox:
            pen = QPen(QColor(255, 165, 0))  # Orange color
            pen.setStyle(Qt.DashLine)
            pen.setWidth(2)
            self.bbox_item = self.addRect(bbox[0], bbox[1], bbox[2], bbox[3], pen)


class IntegratedPoseTool(QMainWindow):
    def __init__(self, pose_config):
        super().__init__()
        self.pose_config = pose_config  # Store the pose config
        self.video_processor = VideoProcessor()
        self.current_frame_number = 0
        self.output_dir = None
        self.current_working_image = None  # To track if we are working on video or annotation
        self.annotations = self.create_empty_annotations()
        self.initUI()

    def create_empty_annotations(self):
        return {
            "info": {
                "description": "Pose Keypoint Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"url": "", "id": 1, "name": ""}],
            "images": [],
            "annotations": [],
            "categories": [self.pose_config.get_category_config()]
        }
        
    def setOutputDirectory(self):
        self.output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if self.output_dir:
            # Create necessary subdirectories
            os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
            
            # Check for existing annotations
            annotation_file = os.path.join(self.output_dir, 'annotations.json')
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r') as f:
                        self.annotations = json.load(f)
                    # Update frame dropdown with existing annotations
                    self.updateFrameDropdown()
                    QMessageBox.information(self, "Loaded Annotations", 
                                        f"Loaded existing annotations from:\n{annotation_file}\n"
                                        f"Contains {len(self.annotations['images'])} images and "
                                        f"{len(self.annotations['annotations'])} annotations.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load existing annotations: {str(e)}")
            else:
                QMessageBox.information(self, "New Annotations", 
                                    f"Will create new annotations file at:\n{annotation_file}")
    
        
    def exitProgram(self):
        reply = QMessageBox.question(self, 'Exit Program',
                                   'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.close()

    def initUI(self):
        self.setWindowTitle('Integrated Pose Annotation & Visualization Tool')
        self.setGeometry(100, 100, 1400, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create image viewer with pose config
        self.viewer = ImageViewer(self.pose_config)
        layout.addWidget(self.viewer, stretch=2)
        
        # Create right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        layout.addWidget(right_panel, stretch=1)
        
        # File controls section
        file_group = QVBoxLayout()
        load_video_btn = QPushButton('Load Video')
        load_video_btn.clicked.connect(self.loadVideo)
        file_group.addWidget(load_video_btn)
        
        load_annotations_btn = QPushButton('Load Annotations')
        load_annotations_btn.clicked.connect(self.loadAnnotations)
        file_group.addWidget(load_annotations_btn)
        
        set_output_btn = QPushButton('Set Output Directory')
        set_output_btn.clicked.connect(self.setOutputDirectory)
        file_group.addWidget(set_output_btn)
        right_layout.addLayout(file_group)
        
        # Frame selection section
        frame_group = QVBoxLayout()
        right_layout.addWidget(QLabel('Frame Selection:'))
        
        # Dropdown for labeled frames
        self.frame_dropdown = QComboBox()
        self.frame_dropdown.currentIndexChanged.connect(self.loadSelectedFrame)
        frame_group.addWidget(self.frame_dropdown)
        
        # Frame slider for video navigation
        frame_control = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.updateFrame)
        frame_control.addWidget(self.frame_slider)
        
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.valueChanged.connect(self.updateFrame)
        frame_control.addWidget(self.frame_spinbox)
        frame_group.addLayout(frame_control)
        right_layout.addLayout(frame_group)
        
        # Keypoint list and controls
        right_layout.addWidget(QLabel('Keypoints:'))
        self.keypoint_list = QListWidget()
        self.keypoint_list.addItems(self.pose_config.keypoint_names)
        self.keypoint_list.setFixedHeight(
            self.keypoint_list.sizeHintForRow(0) * len(self.pose_config.keypoint_names) + 10)
        self.keypoint_list.currentTextChanged.connect(
            lambda x: self.viewer.scene().set_current_keypoint(x))
        self.keypoint_list.setCurrentRow(0)
        right_layout.addWidget(self.keypoint_list)
        
        # Control buttons
        buttons_layout = QVBoxLayout()
        reset_keypoint_btn = QPushButton('Reset Selected Keypoint')
        reset_keypoint_btn.clicked.connect(self.resetSelectedKeypoint)
        buttons_layout.addWidget(reset_keypoint_btn)

        save_btn = QPushButton('Save Current Frame')
        save_btn.clicked.connect(self.saveBtnClicked)  
        buttons_layout.addWidget(save_btn)
        
        reset_btn = QPushButton('Reset All Keypoints')
        reset_btn.clicked.connect(self.resetCurrent)
        buttons_layout.addWidget(reset_btn)
        
        # Add exit button
        exit_btn = QPushButton('Exit Program')
        exit_btn.clicked.connect(self.exitProgram)
        buttons_layout.addWidget(exit_btn)
        
        right_layout.addLayout(buttons_layout)
        
        # Metadata display
        self.info_label = QLabel()
        right_layout.addWidget(self.info_label)
        
        # Set up keypoint update callback
        self.viewer.scene().keypoint_updated = self.updateKeypointStatus
        
        # Message prompt region
        right_layout.addWidget(QLabel('Status Messages:'))
        self.message_prompt = QTextEdit()
        self.message_prompt.setReadOnly(True)  # Make it read-only
        self.message_prompt.setMaximumHeight(100)  # Limit height
        right_layout.addWidget(self.message_prompt)
    
    def saveBtnClicked(self):
        # First, get the current frame info from the metadata display
        source_text = self.info_label.text()
        try:
            video_line = next((line for line in source_text.split('\n') if line.startswith('Video:')), "Video: Unknown")
            frame_line = next((line for line in source_text.split('\n') if line.startswith('Frame:')), "Frame: 0")
            id_line = next((line for line in source_text.split('\n') if line.startswith('Image ID:')), "Image ID: None")
            
            current_video = video_line.split(': ')[1]
            current_frame = frame_line.split(': ')[1]
            current_id = id_line.split(': ')[1]
        except:
            current_video = "Unknown"
            current_frame = "Unknown"
            current_id = "Unknown"
        
        # Call the original save function
        self.saveAnnotations()
        
        # Add status message
        message = f"Saved: Frame {current_frame} (ID: {current_id}) from {current_video}"
        self.addStatusMessage(message, "red")
    
    
    def addStatusMessage(self, message, color="black"):
        # Get current time
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Format message with timestamp
        formatted_message = f"[{current_time}] {message}"
        
        # Create HTML with specified color
        html = f"<span style='color:{color};'>{formatted_message}</span><br>"
        
        # Add message to the prompt
        self.message_prompt.moveCursor(QTextCursor.End)
        self.message_prompt.insertHtml(html)
        self.message_prompt.ensureCursorVisible()       
        
    def updateKeypointStatus(self, keypoint_name, is_labeled):
        items = self.keypoint_list.findItems(keypoint_name, Qt.MatchExactly)
        if items:
            item = items[0]
            # Get the keypoint's visibility value (v) from the scene
            visibility = 0  # default - not labeled
            if keypoint_name in self.viewer.scene().keypoints:
                _, _, v = self.viewer.scene().keypoints[keypoint_name]
                visibility = v
                
            if is_labeled:
                if visibility == 1:  # not visible but labeled
                    item.setBackground(QColor(255, 255, 0))  # Yellow for not visible
                else:  # visibility == 2, visible
                    item.setBackground(QColor(200, 255, 200))  # Light green for visible
            else:
                item.setBackground(QColor(255, 255, 255))  # White for unlabeled

    # NEW: Enhanced load annotations method
    def loadAnnotations(self):
        annotations_file, _ = QFileDialog.getOpenFileName(
            self, "Select Annotations File", "", "JSON Files (*.json)")
        
        if not annotations_file:
            return
            
        try:
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
            
            # Set output directory to annotations location
            self.output_dir = os.path.dirname(annotations_file)
            
            # Update frame dropdown
            self.updateFrameDropdown()
            
            QMessageBox.information(self, "Loaded Annotations", 
                                  f"Successfully loaded {len(self.annotations['images'])} "
                                  f"images and {len(self.annotations['annotations'])} annotations.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load annotations: {str(e)}")
            
    def updateFrameDropdown(self):
        self.frame_dropdown.clear()
        for image in self.annotations['images']:
            self.frame_dropdown.addItem(
                f"Frame {image['frame_number']} (ID: {image['id']})", 
                userData=image['id'])


    def displayFrame(self, frame, annotation_data=None):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create new scene
        new_scene = KeypointScene(self.pose_config)
        self.viewer.setScene(new_scene)
        
        # Add image to scene
        pixmap = QPixmap.fromImage(q_image)
        new_scene.addPixmap(pixmap)
        self.viewer.setSceneRect(QRectF(pixmap.rect()))
        self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)
        
        # Reset keypoint highlights
        for i in range(self.keypoint_list.count()):
            self.keypoint_list.item(i).setBackground(QColor(255, 255, 255))
        
        # Set up keypoint update callback
        new_scene.keypoint_updated = self.updateKeypointStatus
        
        # Load existing keypoints if provided
        if annotation_data:
            keypoints = annotation_data['keypoints']
            for i, kp_name in enumerate(self.pose_config.keypoint_names):
                x = keypoints[i * 3]
                y = keypoints[i * 3 + 1]
                v = keypoints[i * 3 + 2]
                if v > 0:  # If keypoint exists
                    new_scene.keypoints[kp_name] = (x, y, v)
                    self.updateKeypointStatus(kp_name, True)
            
            new_scene.update_keypoint_visuals()
        
        # Preserve the selected keypoint
        current_keypoint = self.keypoint_list.currentItem().text()
        new_scene.set_current_keypoint(current_keypoint)

    def updateMetadataDisplay(self, image_data, annotation_data):
        bbox = annotation_data.get('bbox', [0, 0, 0, 0])
        
        # Determine the source
        if image_data.get('id') is None:
            source = "Video only (not annotated)"
        else:
            if hasattr(self.video_processor, 'video_file') and self.video_processor.video_file == image_data.get('video_file'):
                source = "Annotation and Video"
            else:
                source = "Annotation only"
        
        # Get visibility counts
        visible_points = len([k for k in annotation_data['keypoints'][2::3] if k == 2])
        estimated_points = len([k for k in annotation_data['keypoints'][2::3] if k == 1])
        
        info_text = (
            f"Source: {source}\n"
            f"Video: {image_data.get('video_file', 'N/A')}\n"
            f"Frame: {image_data['frame_number']}\n"
            f"Image ID: {image_data.get('id', 'N/A')}\n"
            f"BBox: x={bbox[0]:.1f}, y={bbox[1]:.1f}, "
            f"w={bbox[2]:.1f}, h={bbox[3]:.1f}\n"
            f"Visible Keypoints (Left-click): {visible_points}\n"
            f"Estimated Keypoints (Right-click): {estimated_points}\n"
            f"Unlabeled Keypoints: {len(self.pose_config.keypoint_names) - visible_points - estimated_points}"
        )
        self.info_label.setText(info_text)
        
    def loadVideo(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_processor.video_file = os.path.basename(video_path)
            self.video_processor.load_video(video_path)
            self.frame_slider.setMaximum(self.video_processor.total_frames - 1)
            self.frame_spinbox.setMaximum(self.video_processor.total_frames - 1)
            self.updateFrame(0)

    def loadSelectedFrame(self, index):
        if index < 0:
            return
            
        image_id = self.frame_dropdown.currentData()
        image_data = next(img for img in self.annotations['images'] 
                         if img['id'] == image_id)
        
        # Load corresponding annotation
        annotation_data = next(ann for ann in self.annotations['annotations'] 
                             if ann['image_id'] == image_id)
        
        # Sync video frame if the video matches
        if (hasattr(self.video_processor, 'video_file') and 
            self.video_processor.video_file == image_data['video_file']):
            self.current_frame_number = image_data['frame_number']
            self.frame_slider.setValue(self.current_frame_number)
            self.frame_spinbox.setValue(self.current_frame_number)
            
            # Get frame from video
            frame = self.video_processor.get_frame(self.current_frame_number)
        else:
            # Load from saved frame
            image_path = os.path.join(self.output_dir, "frames", image_data['file_name'])
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "Error", f"Image file not found: {image_path}")
                return
            frame = cv2.imread(image_path)
        
        if frame is None:
            return
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.displayFrame(frame, annotation_data)
        self.updateMetadataDisplay(image_data, annotation_data)
    
    def updateFrame(self, frame_number):
        self.current_frame_number = frame_number
        self.frame_slider.setValue(frame_number)
        self.frame_spinbox.setValue(frame_number)
        
        frame = self.video_processor.get_frame(frame_number)
        if frame is not None:
            # Check if this frame is already annotated
            existing_annotation = None
            existing_image = None
            if hasattr(self.video_processor, 'video_file'):
                for img in self.annotations["images"]:
                    if (img["video_file"] == self.video_processor.video_file and 
                        img["frame_number"] == frame_number):
                        existing_image = img
                        existing_annotation = next(
                            ann for ann in self.annotations["annotations"] 
                            if ann["image_id"] == img["id"])
                        break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.displayFrame(frame, existing_annotation)
            
            if existing_image and existing_annotation:
                self.updateMetadataDisplay(existing_image, existing_annotation)
            else:
                # Create temporary image data for video-only frame
                temp_image_data = {
                    "video_file": getattr(self.video_processor, 'video_file', 'N/A'),
                    "frame_number": frame_number,
                    "id": None
                }
                temp_annotation_data = {
                    "bbox": [0, 0, 0, 0],
                    "keypoints": [0] * (len(self.pose_config.keypoint_names) * 3)
                }
                self.updateMetadataDisplay(temp_image_data, temp_annotation_data)
    
    def saveAnnotations(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please set output directory first!")
            return
        
        # Check for existing annotation by matching image data from metadata
        source_text = self.info_label.text()
        # Parse video and frame from metadata
        video_line = next(line for line in source_text.split('\n') if line.startswith('Video:'))
        frame_line = next(line for line in source_text.split('\n') if line.startswith('Frame:'))
        current_video = video_line.split(': ')[1]
        current_frame = int(frame_line.split(': ')[1])
        
        existing_annotation = None
        existing_image = None
        
        for i, img in enumerate(self.annotations["images"]):
            if (img["video_file"] == current_video and 
                img["frame_number"] == current_frame):
                existing_image = img
                existing_annotation = next(
                    (ann for ann in self.annotations["annotations"] 
                    if ann["image_id"] == img["id"]), None)
                break
                    
        if existing_annotation:
            reply = QMessageBox.question(self, 'Duplicate Frame',
                                       f'Frame {current_frame} from video '
                                       f'"{current_video}" already exists. '
                                       'Do you want to update it?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # Update existing annotation with current scene keypoints
                image_id = existing_image["id"]
                
                # Prepare keypoints from current scene
                keypoints = []
                for kp_name in self.pose_config.keypoint_names:
                    if kp_name in self.viewer.scene().keypoints:
                        x, y, v = self.viewer.scene().keypoints[kp_name]
                        keypoints.extend([x, y, v])
                    else:
                        keypoints.extend([0, 0, 0])
                
                # Calculate bbox from current scene
                bbox = self.viewer.scene().calculate_bbox() or [0, 0, 0, 0]
                area = bbox[2] * bbox[3] if bbox else 0
                
                # Update existing annotation
                existing_annotation.update({
                    "keypoints": keypoints,
                    "num_keypoints": len(self.viewer.scene().keypoints),
                    "bbox": bbox,
                    "area": area
                })
                
                # Update image info timestamp
                existing_image["date_captured"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Save to file
                with open(os.path.join(self.output_dir, 'annotations.json'), 'w') as f:
                    json.dump(self.annotations, f, indent=2)
                
                QMessageBox.information(self, "Success", 
                                      f"Frame {current_frame} updated successfully!")
                return
            else:
                return
        
        # If not updating existing annotation, proceed with new annotation...
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # For new annotation, get next available ID
        image_id = max([img["id"] for img in self.annotations["images"]], default=0) + 1
        
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save current frame
        filename = self.video_processor.save_frame(self.current_frame_number, frames_dir, image_id)
        
        if filename:
            # Create image info
            image_info = {
                "id": image_id,
                "file_name": filename,
                "video_file": self.video_processor.video_file,
                "frame_number": self.current_frame_number,
                "width": self.video_processor.frame_width,
                "height": self.video_processor.frame_height,
                "fps": self.video_processor.fps,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Prepare keypoints
            keypoints = []
            for kp_name in self.pose_config.keypoint_names:
                if kp_name in self.viewer.scene().keypoints:
                    x, y, v = self.viewer.scene().keypoints[kp_name]
                    keypoints.extend([x, y, v])
                else:
                    keypoints.extend([0, 0, 0])
            
            # Calculate bbox
            bbox = self.viewer.scene().calculate_bbox() or [0, 0, 0, 0]
            area = bbox[2] * bbox[3] if bbox else 0
            
            # Create annotation
            annotation = {
                "id": len(self.annotations["annotations"]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": len(self.viewer.scene().keypoints),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
                "score": 1.0
            }
            
            # Update annotations
            self.annotations["images"].append(image_info)
            self.annotations["annotations"].append(annotation)
            
            # Save to file
            with open(os.path.join(self.output_dir, 'annotations.json'), 'w') as f:
                json.dump(self.annotations, f, indent=2)
            
            
            # Update frame dropdown
            self.updateFrameDropdown()
            
            # Refresh the display
            # Get the newly created/updated image ID
            new_image_id = image_id  # This is already set for both new and existing annotations
            
            # Find the index in dropdown for this frame
            for i in range(self.frame_dropdown.count()):
                if self.frame_dropdown.itemData(i) == new_image_id:
                    # Block signals temporarily to avoid triggering loadSelectedFrame twice
                    self.frame_dropdown.blockSignals(True)
                    self.frame_dropdown.setCurrentIndex(i)
                    self.frame_dropdown.blockSignals(False)
                    # Manually refresh the frame after setting index
                    self.loadSelectedFrame(i)
                    break
    
        
            QMessageBox.information(self, "Success", 
                                  f"Frame {current_frame} saved successfully!")
            
            self.frame_dropdown.setCurrentIndex(i)
            

    def resetSelectedKeypoint(self):
        current_keypoint = self.keypoint_list.currentItem().text()
        self.viewer.scene().reset_keypoint(current_keypoint)
        
    def resetCurrent(self):
        self.viewer.scene().keypoints.clear()
        self.viewer.scene().update_keypoint_visuals()
        for i in range(self.keypoint_list.count()):
            self.keypoint_list.item(i).setBackground(QColor(255, 255, 255))
            
    def closeEvent(self, event):
        self.video_processor.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pose_config=PoseConfig()
    tool = IntegratedPoseTool(pose_config)
    tool.show()
    sys.exit(app.exec_())