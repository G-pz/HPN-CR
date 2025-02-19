import os
import shutil

# Transfer all data from the subfolders of the source folder to another folder
# The source folder contains subfolders, each subfolder contains image files
# The target folder will contain all image files from the source folder

#Target folder path
determination = ['./SEN12MS-CR/train/s1/', './SEN12MS-CR/train/s2_cloudfree/', './SEN12MS-CR/train/s2_cloudy/', 
                 './SEN12MS-CR/val/s1/', './SEN12MS-CR/val/s2_cloudfree/', './SEN12MS-CR/val/s2_cloudy/',
                 './SEN12MS-CR/test/s1/', './SEN12MS-CR/test/s2_cloudfree/', './SEN12MS-CR/test/s2_cloudy/'] 

#Source folder path
path = ['./SEN12MS-CR/train/source_folder/s1/', './SEN12MS-CR/train/source_folder/s2/', './SEN12MS-CR/train/source_folder/s2_cloudy/', 
        './SEN12MS-CR/val/source_folder/s1/', './SEN12MS-CR/val/source_folder/s2/', './SEN12MS-CR/val/source_folder/s2_cloudy/', 
        './SEN12MS-CR/test/source_folder/s1/', './SEN12MS-CR/test/source_folder/s2/', './SEN12MS-CR/test/source_folder/s2_cloudy/']

for i in range(len(determination)):
    determination = determination[i]
    path = path[i]
    
    if not os.path.exists(determination):
        os.makedirs(determination)

    folders = os.listdir(path)
    for folder in folders:
        dir = path + '/' + str(folder)
        files = os.listdir(dir)
        for file in files:
            source = dir + '/' + str(file)
            deter = determination + '/' + str(file)
            shutil.move(source, deter)
