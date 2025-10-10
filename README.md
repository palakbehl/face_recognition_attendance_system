# Fast Face Recognition Attendance System

An optimized face recognition attendance system with multiple speed options.

## Versions (Fastest to Slowest)

1. **Ultra Fast** (`ultra_fast_face.py`) - Maximum speed optimization (~180 lines)
2. **Fast** (`fast_face_recognition.py`) - Balanced speed and accuracy (~190 lines)
3. **Minimal** (`minimal_face.py`) - Basic functionality (~115 lines)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run any version:**
   ```bash
   # Ultra Fast version (fastest)
   python ultra_fast_face.py
   
   # Fast version (balanced)
   python fast_face_recognition.py
   
   # Minimal version (basic)
   python minimal_face.py
   ```

## Performance Comparison

| Version | Recognition Speed | Accuracy | Code Size |
|---------|------------------|----------|-----------|
| Ultra Fast | ⚡⚡⚡ Fastest | ★★★☆☆ | ~180 lines |
| Fast | ⚡⚡ Fast | ★★★★☆ | ~190 lines |
| Minimal | ⚡ Normal | ★★★★☆ | ~115 lines |

## Features

- Real-time face recognition
- Attendance logging with cooldown
- Simple text-based interface
- SQLite database storage
- Multiple speed/accuracy options

## Dependencies

- `opencv-python` - Computer vision
- `numpy` - Numerical computations

Total project: **3 versions + 2 dependencies**

## Usage

### Original Version
1. **Add Person**: Click "Add Person", enter name, capture face images with SPACE key
2. **Import Images**: Click "Import Images" to add faces from files
3. **Start Recognition**: Click "Start Recognition" to begin attendance tracking
4. **View Records**: Click "View Records" to see attendance history
5. **Export**: Click "Export Attendance" to save records as CSV

### Console Version
1. Run `python simple_face_recognition.py`
2. Use the menu to add people, start recognition, etc.
3. Follow on-screen instructions

### Minimal GUI Version
1. Run `python minimal_face_gui.py`
2. Use the clean interface with essential functions only

## Project Structure

```
simple-face-recognition/
├── face_recognition_app.py    # Original full application
├── simple_face_recognition.py # Console version
├── minimal_face_gui.py        # Minimal GUI version
├── ultra_simple_face.py       # Ultra simple version
├── minimal_face.py            # Minimal version
├── requirements.txt           # 4 essential dependencies
├── README.md                 # This file
└── data/                     # Auto-created data storage
    ├── faces/                # Face images
    ├── face_data.json        # Face encodings
    └── attendance.db         # SQLite database
```

## Improvements Made

- **Better Accuracy**: Uses LBP (Local Binary Pattern) + Histogram + Gradient features
- **Optimized Detection**: Improved face detection parameters
- **Single File**: All functionality in one file for maximum simplicity
- **Minimal Dependencies**: Only 4 essential packages
- **Better Recognition**: Higher threshold and multiple feature comparison

## Dependencies

- `opencv-python` - Computer vision
- `numpy` - Numerical computations  
- `pandas` - Data handling
- `Pillow` - Image processing

Total project: **5 versions + 4 dependencies**

# Minimal Face Recognition Attendance System

An ultra-minimal face recognition attendance system with only essential functionality.

## Project Structure

This project includes only the essential files:
1. **Minimal Version** (`minimal_face.py`) - Bare essentials only (~115 lines)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python minimal_face.py
   ```

## Usage

1. Run `python minimal_face.py`
2. Use the menu to add people, start recognition, etc.
3. Follow on-screen instructions

## Features

- Ultra-minimal codebase (only ~115 lines)
- Real-time face recognition
- Attendance logging with cooldown
- Simple text-based interface
- SQLite database storage

## Dependencies

- `opencv-python` - Computer vision
- `numpy` - Numerical computations

Total project: **1 version + 2 dependencies**

# Face Recognition Attendance System

A simple and robust face recognition system for attendance tracking.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python face_recognition.py
   ```

## Usage

1. **Add Person**: Select option 1 to add a person's face to the system
   - Enter the person's name when prompted
   - Position your face in the camera view
   - Press SPACE to capture the face
   - Press ESC to cancel

2. **Start Recognition**: Select option 2 to start recognizing faces
   - The camera will detect and recognize faces
   - Recognized faces will be shown with green boxes
   - Unknown faces will be shown with red boxes
   - Attendance is automatically logged (once per 5 minutes per person)
   - Press ESC to stop recognition

3. **View Records**: Select option 3 to view attendance records
   - Shows the 20 most recent attendance entries

4. **Exit**: Select option 4 to exit the application

## Features

- Simple and robust face recognition
- Attendance tracking with cooldown period
- SQLite database for storing records
- Error handling for common issues

## Dependencies

- `opencv-python` - Computer vision
- `numpy` - Numerical computations

Total project: **1 version + 2 dependencies**
