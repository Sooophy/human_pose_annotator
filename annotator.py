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
from PyQt5.QtCore import Qt, QRectF

from pose_config import*


class VideoProcessor:
    def __init__(self):
        self.video_path = None
        self.video_file = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        
    def load_video(self, video_path):
        if self.cap is not None:
            self.cap.release()

        self.video_path = video_path
        self.video_file = os.path.basename(video_path)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.total_frames = 0
            self.fps = 0
            self.frame_width = 0
            self.frame_height = 0
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.total_frames <= 0:
            self.cap.release()
            self.cap = None
            return False
        return True

    def frame_range(self):
        if self.total_frames <= 0:
            return None
        return 0, self.total_frames - 1
        
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
            cv2.imwrite(output_path, frame)
            return filename
        return None
    
    def close(self):
        if self.cap is not None:
            self.cap.release()


class ImageFolderProcessor:
    valid_extensions = {".jpg", ".jpeg", ".png"}

    def __init__(self):
        self.folder_path = None
        self.video_file = None
        self.frame_paths = {}
        self.frame_numbers = []
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0

    def load_folder(self, folder_path):
        self.folder_path = folder_path
        self.video_file = os.path.basename(os.path.normpath(folder_path))
        self.frame_paths = {}
        self.frame_numbers = []
        self.frame_width = 0
        self.frame_height = 0

        for filename in os.listdir(folder_path):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in self.valid_extensions or not stem.isdigit():
                continue
            frame_number = int(stem)
            self.frame_paths[frame_number] = os.path.join(folder_path, filename)

        self.frame_numbers = sorted(self.frame_paths)
        if not self.frame_numbers:
            return False

        first_frame = self.get_frame(self.frame_numbers[0])
        if first_frame is None:
            return False

        self.frame_height, self.frame_width = first_frame.shape[:2]
        return True

    def frame_range(self):
        if not self.frame_numbers:
            return None
        return self.frame_numbers[0], self.frame_numbers[-1]

    def get_frame(self, frame_number):
        image_path = self.frame_paths.get(frame_number)
        if image_path is None:
            return None
        return cv2.imread(image_path)

    def save_frame(self, frame_number, output_dir, image_id):
        frame = self.get_frame(frame_number)
        if frame is not None:
            filename = f"{image_id:012d}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            return filename
        return None

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
                color = QColor(base_color)
            
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
        self.image_folder_processor = ImageFolderProcessor()
        self.active_source = None
        self.current_frame_number = 0
        self.output_dir = None
        self.current_image_data = None
        self.current_annotation_data = None
        self.current_frame_bgr = None
        self.is_syncing_frame_controls = False
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
                    self.normalizeAnnotations()
                    # Update frame dropdown with existing annotations
                    self.updateFrameDropdown()
                    QMessageBox.information(self, "Loaded Annotations", 
                                        f"Loaded existing annotations from:\n{annotation_file}\n"
                                        f"Contains {len(self.annotations.get('images', []))} images and "
                                        f"{len(self.annotations.get('annotations', []))} annotations.")
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

        load_image_folder_btn = QPushButton('Load Image Folder')
        load_image_folder_btn.clicked.connect(self.loadImageFolder)
        file_group.addWidget(load_image_folder_btn)
        
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
        image_data = self.saveAnnotations()
        if not image_data:
            return

        current_video = image_data.get("video_file", "Unknown")
        current_frame = image_data.get("frame_number", "Unknown")
        current_id = image_data.get("id", "Unknown")
        
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

    def find_image(self, image_id):
        return next((img for img in self.annotations.get('images', [])
                     if img.get('id') == image_id), None)

    def find_annotation(self, image_id):
        return next((ann for ann in self.annotations.get('annotations', [])
                     if ann.get('image_id') == image_id), None)

    def find_existing_frame(self, video_file, frame_number):
        for image in self.annotations.get("images", []):
            if (image.get("video_file") == video_file and
                    image.get("frame_number") == frame_number):
                return image, self.find_annotation(image.get("id"))
        return None, None

    def build_keypoints(self):
        keypoints = []
        for kp_name in self.pose_config.keypoint_names:
            if kp_name in self.viewer.scene().keypoints:
                x, y, v = self.viewer.scene().keypoints[kp_name]
                keypoints.extend([x, y, v])
            else:
                keypoints.extend([0, 0, 0])
        return keypoints

    def next_annotation_id(self):
        return max([ann.get("id", 0) for ann in self.annotations.get("annotations", [])],
                   default=0) + 1

    def setFrameControls(self, frame_number):
        self.is_syncing_frame_controls = True
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        self.frame_slider.setValue(frame_number)
        self.frame_spinbox.setValue(frame_number)
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)
        self.is_syncing_frame_controls = False

    def setFrameRange(self, minimum, maximum):
        self.is_syncing_frame_controls = True
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        self.frame_slider.setMinimum(minimum)
        self.frame_spinbox.setMinimum(minimum)
        self.frame_slider.setMaximum(maximum)
        self.frame_spinbox.setMaximum(maximum)
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)
        self.is_syncing_frame_controls = False

    def setCurrentFrameState(self, image_data, annotation_data, frame_bgr):
        self.current_image_data = image_data
        self.current_annotation_data = annotation_data
        self.current_frame_bgr = frame_bgr
        self.current_frame_number = image_data.get("frame_number", 0)

    def active_source_matches(self, image_data):
        return self.active_source is not None and self.active_source.video_file == image_data.get("video_file")

    def activateSource(self, source):
        self.active_source = source
        min_frame, max_frame = source.frame_range()
        self.setFrameRange(min_frame, max_frame)
        self.updateFrame(min_frame)

    def normalizeAnnotations(self):
        self.annotations.setdefault('images', [])
        self.annotations.setdefault('annotations', [])
        self.annotations.setdefault('categories', [self.pose_config.get_category_config()])

    def writeAnnotationsFile(self):
        with open(os.path.join(self.output_dir, 'annotations.json'), 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def selectFrameDropdownByImageId(self, image_id):
        for i in range(self.frame_dropdown.count()):
            if self.frame_dropdown.itemData(i) == image_id:
                self.frame_dropdown.blockSignals(True)
                self.frame_dropdown.setCurrentIndex(i)
                self.frame_dropdown.blockSignals(False)
                return i
        return -1

    def showFrameState(self, image_data, annotation_data, frame_bgr):
        self.setCurrentFrameState(image_data, annotation_data, frame_bgr)
        self.displayFrame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), annotation_data)
        self.updateMetadataDisplay(image_data, annotation_data)

    # NEW: Enhanced load annotations method
    def loadAnnotations(self):
        annotations_file, _ = QFileDialog.getOpenFileName(
            self, "Select Annotations File", "", "JSON Files (*.json)")
        
        if not annotations_file:
            return
            
        try:
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
            self.normalizeAnnotations()
            
            # Set output directory to annotations location
            self.output_dir = os.path.dirname(annotations_file)
            
            # Update frame dropdown
            self.updateFrameDropdown()
            
            QMessageBox.information(self, "Loaded Annotations", 
                                  f"Successfully loaded {len(self.annotations.get('images', []))} "
                                  f"images and {len(self.annotations.get('annotations', []))} annotations.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load annotations: {str(e)}")
            
    def updateFrameDropdown(self):
        self.frame_dropdown.blockSignals(True)
        self.frame_dropdown.clear()
        for image in self.annotations.get('images', []):
            if 'id' not in image or 'frame_number' not in image:
                continue
            self.frame_dropdown.addItem(
                f"Frame {image['frame_number']} (ID: {image['id']})", 
                userData=image['id'])
        self.frame_dropdown.blockSignals(False)


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
            keypoints = annotation_data.get('keypoints', [])
            for i, kp_name in enumerate(self.pose_config.keypoint_names):
                if i * 3 + 2 >= len(keypoints):
                    break
                x = keypoints[i * 3]
                y = keypoints[i * 3 + 1]
                v = keypoints[i * 3 + 2]
                if v > 0:  # If keypoint exists
                    new_scene.keypoints[kp_name] = (x, y, v)
                    self.updateKeypointStatus(kp_name, True)
            
            new_scene.update_keypoint_visuals()
        
        # Preserve the selected keypoint
        current_item = self.keypoint_list.currentItem()
        if current_item:
            new_scene.set_current_keypoint(current_item.text())

    def updateMetadataDisplay(self, image_data, annotation_data):
        bbox = annotation_data.get('bbox', [0, 0, 0, 0])
        
        # Determine the source
        if image_data.get('id') is None:
            source = "Current source only (not annotated)"
        else:
            if self.active_source_matches(image_data):
                source = "Annotation and Current Source"
            else:
                source = "Annotation only"
        
        # Get visibility counts
        keypoints = annotation_data.get('keypoints', [])
        visible_points = len([k for k in keypoints[2::3] if k == 2])
        estimated_points = len([k for k in keypoints[2::3] if k == 1])
        
        info_text = (
            f"Source: {source}\n"
            f"Video: {image_data.get('video_file', 'N/A')}\n"
            f"Frame: {image_data.get('frame_number', 'N/A')}\n"
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
            if not self.video_processor.load_video(video_path):
                QMessageBox.warning(self, "Error", f"Failed to load video or video has no frames:\n{video_path}")
                return

            self.activateSource(self.video_processor)

    def loadImageFolder(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Any Image in the Folder",
            "",
            "Image Files (*.jpg *.jpeg *.png)"
        )
        if not image_path:
            return

        folder_path = os.path.dirname(image_path)
        if not self.image_folder_processor.load_folder(folder_path):
            QMessageBox.warning(
                self,
                "Error",
                "No readable numbered images found in the selected folder.\n"
                "Expected names like 000000000000.jpg, 000000000001.jpg, ..."
            )
            return

        self.video_processor.close()
        self.activateSource(self.image_folder_processor)

    def loadSelectedFrame(self, index):
        if index < 0:
            return
            
        image_id = self.frame_dropdown.currentData()
        image_data = self.find_image(image_id)
        if image_data is None:
            QMessageBox.warning(self, "Error", f"Annotation image ID not found: {image_id}")
            return
        
        # Load corresponding annotation
        annotation_data = self.find_annotation(image_id)
        if annotation_data is None:
            QMessageBox.warning(self, "Error", f"Annotation data not found for image ID: {image_id}")
            return
        
        # Sync active source frame if it matches the annotation source.
        if self.active_source_matches(image_data):
            self.current_frame_number = image_data.get('frame_number', 0)
            self.setFrameControls(self.current_frame_number)
            frame = self.active_source.get_frame(self.current_frame_number)
        else:
            # Load from saved frame
            file_name = image_data.get('file_name')
            if not file_name:
                QMessageBox.warning(self, "Error", f"Image file name missing for image ID: {image_id}")
                return
            image_path = os.path.join(self.output_dir, "frames", file_name)
            if not os.path.exists(image_path):
                QMessageBox.warning(self, "Error", f"Image file not found: {image_path}")
                return
            frame = cv2.imread(image_path)
        
        if frame is None:
            QMessageBox.warning(self, "Error", "Failed to load frame image.")
            return

        self.showFrameState(image_data, annotation_data, frame)
    
    def updateFrame(self, frame_number):
        if self.is_syncing_frame_controls:
            return
        if self.active_source is None:
            return

        frame = self.active_source.get_frame(frame_number)
        if frame is None:
            if self.active_source is self.image_folder_processor:
                self.addStatusMessage(f"Frame {frame_number} is missing from the selected image folder.", "red")
            return

        self.current_frame_number = frame_number
        self.setFrameControls(frame_number)

        source_name = self.active_source.video_file
        existing_image, existing_annotation = self.find_existing_frame(source_name, frame_number)
        if existing_image and existing_annotation:
            self.showFrameState(existing_image, existing_annotation, frame)
        else:
            temp_image_data = {
                "video_file": source_name or 'N/A',
                "frame_number": frame_number,
                "id": None
            }
            temp_annotation_data = {
                "bbox": [0, 0, 0, 0],
                "keypoints": [0] * (len(self.pose_config.keypoint_names) * 3)
            }
            self.showFrameState(temp_image_data, temp_annotation_data, frame)
    
    def saveAnnotations(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please set output directory first!")
            return False

        if self.current_image_data is None or self.current_frame_bgr is None:
            QMessageBox.warning(self, "Warning", "Please load a video frame or annotation frame first!")
            return False
        
        current_video = self.current_image_data.get("video_file")
        current_frame = self.current_image_data.get("frame_number")
        if current_video in (None, "N/A") or current_frame is None:
            QMessageBox.warning(self, "Warning", "Current frame does not have video metadata to save.")
            return False
        
        existing_annotation = None
        existing_image = None
        existing_image, existing_annotation = self.find_existing_frame(current_video, current_frame)
        if existing_image and not existing_annotation:
            QMessageBox.warning(
                self,
                "Warning",
                f"Frame {current_frame} from video \"{current_video}\" has image metadata "
                "but no matching annotation. Please repair the annotation file before saving."
            )
            return False
                    
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
                keypoints = self.build_keypoints()
                
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
                
                self.writeAnnotationsFile()

                self.setCurrentFrameState(existing_image, existing_annotation, self.current_frame_bgr)
                self.updateFrameDropdown()
                self.selectFrameDropdownByImageId(image_id)
                self.updateMetadataDisplay(existing_image, existing_annotation)
                
                QMessageBox.information(self, "Success", 
                                      f"Frame {current_frame} updated successfully!")
                return existing_image
            else:
                return False
        
        # If not updating existing annotation, proceed with new annotation...
        if self.active_source is None:
            QMessageBox.warning(self, "Warning", "Please load a video or image folder before saving a new annotation.")
            return False

        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # For new annotation, get next available ID
        image_id = max([img.get("id", 0) for img in self.annotations.get("images", [])],
                       default=0) + 1
        
        # Save current frame
        filename = self.active_source.save_frame(current_frame, frames_dir, image_id)
        
        if filename:
            # Create image info
            image_info = {
                "id": image_id,
                "file_name": filename,
                "video_file": current_video,
                "frame_number": current_frame,
                "width": self.active_source.frame_width,
                "height": self.active_source.frame_height,
                "fps": self.active_source.fps,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Prepare keypoints
            keypoints = self.build_keypoints()
            
            # Calculate bbox
            bbox = self.viewer.scene().calculate_bbox() or [0, 0, 0, 0]
            area = bbox[2] * bbox[3] if bbox else 0
            
            # Create annotation
            annotation = {
                "id": self.next_annotation_id(),
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
            self.current_image_data = image_info
            self.current_annotation_data = annotation
            
            self.writeAnnotationsFile()
            
            
            # Update frame dropdown
            self.updateFrameDropdown()
            
            # Refresh the display
            # Get the newly created/updated image ID
            selected_index = self.selectFrameDropdownByImageId(image_id)
            if selected_index >= 0:
                self.loadSelectedFrame(selected_index)
    
        
            QMessageBox.information(self, "Success", 
                                  f"Frame {current_frame} saved successfully!")
            
            return image_info
        else:
            QMessageBox.warning(self, "Error", "Failed to save current frame image.")
            return False
            

    def resetSelectedKeypoint(self):
        current_item = self.keypoint_list.currentItem()
        if current_item:
            self.viewer.scene().reset_keypoint(current_item.text())
        
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
