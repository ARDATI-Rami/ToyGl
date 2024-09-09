# macros set_color.py
from paraview.simple import *  # Import ParaView Python functions
import random  # Import random module to generate random colors

# Step 1: Get the active source (your loaded cell)
active_source = GetActiveSource()

# Step 2: Set the representation to Wireframe
display = Show(active_source)
display.Representation = 'Wireframe'

# Step 3: Check if the object is an adhesion (based on its name) or not
if "adhesion" in active_source.GetLogName().lower():
    # Apply red color and thicker line width for adhesion objects
    display.ColorArrayName = [None, '']  # Disable coloring by data array

    display.DiffuseColor = [1.0, 0.0, 0.0]  # Red color
    display.AmbientColor = [1.0, 0.0, 0.0]  # Match ambient color to red
    display.LineWidth = 2  # Thicker line width for adhesions
    print(f"Setting object {active_source.GetLogName()} to color [1.0, 0.0, 0.0] (Red) with LineWidth 2")
else:
    # Apply random color for other objects
    random_color = [random.random(), random.random(), random.random()]
    display.ColorArrayName = [None, '']  # Disable coloring by data array
    display.DiffuseColor = random_color  # Set random color
    display.AmbientColor = random_color  # Match ambient color to the diffuse color
    display.LineWidth = 1  # Default line width
    print(f"Setting object {active_source.GetLogName()} to color {random_color} with LineWidth 1")

# Step 4: Render the scene to apply the changes
Render()
