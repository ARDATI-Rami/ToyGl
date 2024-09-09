# macros set_view.py
from paraview.simple import *


def set_paraview_view_black(render_view):
    """
    Set the view properties for a ParaView render view:
    - Set background color to black.
    - Remove center axes, orientation axes, and axes grid.
    - Disable all scalar bars and labels.

    Args:
        render_view (RenderView): The ParaView render view object.
    """
    # Set the background color to black
    render_view.Background = [0.0, 0.0, 0.0]

    # Remove the center axes, orientation axes, and axes grid
    render_view.CenterAxesVisibility = 0
    render_view.OrientationAxesVisibility = 0
    render_view.AxesGrid.Visibility = 0

    # Ensure no color palette is used for the background
    SetViewProperties(
        Background=[0.0, 0.0, 0.0],  # Black background
        UseColorPaletteForBackground=0
    )

    # Iterate over all the sources and their display properties
    for name, source in GetSources().items():
        print(f"Checking source: {name}")
        display = GetDisplayProperties(source, view=render_view)

        # Disable scalar bars if found
        if hasattr(display, 'ScalarBarVisibility'):
            print(f"Disabling ScalarBarVisibility for {name}")
            display.ScalarBarVisibility = 0

        # Hide any potential scalar bars using this fallback method
        if hasattr(display, 'HideScalarBarIfNotNeeded'):
            display.HideScalarBarIfNotNeeded(render_view)

    print("All bars and labels removed.")


# Get the active view
render_view = GetActiveViewOrCreate('RenderView')

# Apply the black background and remove axes/grid
set_paraview_view_black(render_view)

# Render the view to apply the changes
Render()
