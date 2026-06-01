import os
import glob 
"""Remove all files with a specific extension
from the given folder.
Args:
    folder_path,
    extension
Returns:
    None
"""
def RemoveFile(folder_path,extension):
    if not os.path.exists(folder_path):
        print("Folder does not exist.")
        return
    deleted_count = 0
    # Loop through files
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path,file_name)
        
        # Skip folders
        if not os.path.isfile(file_path):
            continue

        # Delete matching extension
        if file_name.lower().endswith(
                extension.lower()):
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted: {file_name}")

    print(f"\nTotal files deleted:s"f"{deleted_count}")
    
    
"""Convert all files from the given folder.
Args:
    folder_path
Returns:
    None
"""
def ConvertFiles(folder_path):
    # # Loop through all files in the directory
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".png"):
    #         # Open the PNG file
    #         png_image = Image.open(os.path.join(directory, filename))

    #         # Convert the PNG file to JPEG format
    #         jpeg_image = png_image.convert('RGB')

    #         # Set the new file name and save the JPEG file
    #         new_filename = os.path.splitext(filename)[0] + ".jpg"
    #         jpeg_image.save(os.path.join(directory, new_filename))

    #         # Delete the original PNG file if you want to
    #         os.remove(os.path.join(directory, filename))
    pass
    

"""Rename all files from the given folder,
with given Name args 
Args:
    folder_path,
    Name 
Returns:
    None
"""
def RenameFiles(folder_path,Name):
   
   files = os.listdir(folder_path)
   cnt = 0
   for file in files:
        src_path = os.path.join(folder_path,file)
       
        # Skip folders
        if not os.path.isfile(src_path):
            continue

        # Get extension
        _, ext = os.path.splitext(file)

        # New file name
        new_file_name = (f"{Name}_{cnt}{ext}")
        dst_path = os.path.join(folder_path,new_file_name)
            
        # Rename
        os.rename(src_path,dst_path)
        print(f"Renamed: {file} " f"-> {new_file_name}")
        cnt += 1

if __name__=="__main__":

    #RemoveFile("G:\\Vian\\240428002",".png") 
    RenameFiles("G:\\Vian\\240428002\\","img")
