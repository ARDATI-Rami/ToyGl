# macros reload.py
from paraview.simple import *

# Get all the sources in the pipeline
all_sources = GetSources()

# Check if there are any sources in the pipeline
if all_sources:
    for key, source in all_sources.items():
        # Print information about the source
        print(f"Source name: {source.SMProxy.GetXMLLabel()}")
        file_names = source.FileName if hasattr(source, 'FileName') else "No file names available"
        print(f"File name(s): {file_names}")

        # Reload the files for the source if it has file names
        if hasattr(source, 'FileName'):
            ReloadFiles(source)
            print(f"Reloaded files for source: {source.SMProxy.GetXMLLabel()}")
else:
    print("No sources available in the pipeline.")
