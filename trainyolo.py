from roboflow import Roboflow
rf = Roboflow(api_key="BIJQmOudVSHs95lWH2gk")
project = rf.workspace("traffic3dproj").project("ambulance_detect-iuxup")
dataset = project.version(2).download("yolov5")


