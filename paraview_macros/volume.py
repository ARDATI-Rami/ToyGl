from paraview.simple import *
import os


# Function to calculate volume for a given source
def calculate_volume(source):
    # Try to retrieve a descriptive source name
    if hasattr(source, 'FileName'):
        # Check if FileName is a FileNameProperty and extract the actual file name
        file_name_property = source.GetProperty("FileName")
        if isinstance(file_name_property, paraview.servermanager.FileNameProperty):
            file_name = file_name_property.SMProperty.GetElement(0)
            source_name = os.path.basename(file_name)
        else:
            source_name = source.SMProxy.GetXMLLabel()
    else:
        source_name = source.SMProxy.GetXMLLabel()  # Fallback to proxy label if no file name is available

    # Split the filename at '_step' and take the first part
    object_name = source_name.split('_step')[0]
    # Apply Delaunay 3D filter to create a volume from the surface mesh
    delaunay = Delaunay3D(Input=source)
    delaunay.UpdatePipeline()

    # Apply Integrate Variables filter to calculate the volume of the resulting volume mesh
    integrate_variables = IntegrateVariables(Input=delaunay)
    RenameSource(f"{object_name} volume", integrate_variables)

    integrate_variables.UpdatePipeline()
    # Access the result of the integration
    output = servermanager.Fetch(integrate_variables)

    # Retrieve and print the volume
    if output.GetCellData().HasArray("Volume"):
        volume = output.GetCellData().GetArray("Volume").GetValue(0)
        print(f"Cell volume for {object_name}: {volume}")
    else:
        print(f"Volume not found for source {source_name}. Ensure the source is a valid 3D volume.")


# Get all the sources in the pipeline
all_sources = GetSources()

# Check if there are any sources in the pipeline
if all_sources:
    for key, source in all_sources.items():
        # Print information about the source
        if hasattr(source, 'FileName'):
            file_name_property = source.GetProperty("FileName")
            if isinstance(file_name_property, paraview.servermanager.FileNameProperty):
                file_names = [os.path.basename(file_name_property.SMProperty.GetElement(i)) for i in
                              range(file_name_property.SMProperty.GetNumberOfElements())]
                # print(f"File name(s): {file_names}")

        # Calculate volume for the reloaded source with source name printed
        calculate_volume(source)
else:
    print("No sources available in the pipeline.")
