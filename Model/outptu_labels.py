import numpy as np

# Load the saved classes
classes = np.load("label_encoder_classes.npy")

# Print the mapping
print("Label Encoder Classes and Encoded Values:")
for i, class_name in enumerate(classes):
    print(f"Class: {class_name} -> Encoded as: {i}")

# Print all classes
print("\nAll classes in the label encoder:")
print(classes)