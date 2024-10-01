import os
import pytest
from object_detection import detect_objects 

# Paths to test images
valid_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/input_image.jpg'))
invalid_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/non_existent_image.jpg'))

def test_perform_detection_valid_image():
    """Test object detection with a valid image."""
    result = perform_detection(valid_image_path)  # Call the correct function
    assert result is not None  # Check that the result is not None
    assert os.path.exists(result)  # Check that the output image was created

def test_perform_detection_invalid_image():
    """Test object detection with an invalid image."""
    result = perform_detection(invalid_image_path)  # Call the correct function
    assert result is None  # Check that the result is None

def test_perform_detection_empty_image_path():
    """Test object detection with an empty image path."""
    result = perform_detection("")  # Call the correct function
    assert result is None  # Check that the result is None

def test_perform_detection_directory_path():
    """Test object detection with a directory instead of an image."""
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/'))
    result = perform_detection(directory_path)  # Call the correct function
    assert result is None  # Check that the result is None

def test_perform_detection_invalid_format():
    """Test object detection with an unsupported image format."""
    unsupported_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/unsupported_file.txt'))
    result = perform_detection(unsupported_image_path)  # Call the correct function
    assert result is None  # Check that the result is None

def test_detect_objects():
    """Test the detect_objects function."""
    # Set up test data
    test_image = valid_image_path  # Use the valid image path for the test
    result = detect_objects(test_image)
    
    # Assertions to check if the output is as expected
    assert result is not None  # Ensure that the output is not None
    assert os.path.exists(result)  # Check if the output image was created
