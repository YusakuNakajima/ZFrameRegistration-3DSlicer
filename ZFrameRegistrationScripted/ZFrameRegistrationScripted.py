import os, sys
moduleDir = os.path.dirname(os.path.realpath(__file__))
if moduleDir not in sys.path:
    sys.path.append(moduleDir)

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
from ZFrame.Registration import zf, ZFrameRegistration

class ZFrameRegistrationScripted(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ZFrameRegistration Scripted"
        self.parent.categories = ["IGT"]
        self.parent.dependencies = []
        self.parent.contributors = ["Yusaku Nakajima"]
        self.parent.helpText = """
            This module performs Z-frame registration with selectable orientation.
            """
        self.parent.acknowledgementText = """ """
        for iconExtension in ['.svg', '.png']:
            iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
            if os.path.isfile(iconPath):
                parent.icon = qt.QIcon(iconPath)
                break

class ZFrameRegistrationScriptedWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        if not parent:
            self.parent = slicer.qMRMLWidget()
            self.parent.setLayout(qt.QVBoxLayout())
            self.parent.setMRMLScene(slicer.mrmlScene)
        else:
            self.parent = parent
        self.layout = self.parent.layout()
        if not parent:
            self.setup()
            self.parent.show()

    def onReload(self,moduleName="ZFrameRegistrationScripted"):
        self.zFrameTopologies = {}
        if 'ZFrame.Registration' in sys.modules:
            del sys.modules['ZFrame.Registration']
            print("ZFrame.Registration Deleted")
        globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ZFrameRegistrationScriptedLogic()
        self.zFrameTopologies = {}
        
        # Parameters Area
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
        
        # Input volume selector
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Pick the input volume.")
        parametersFormLayout.addRow("Input Volume: ", self.inputSelector)
        self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeSelected)

        # Orientation Selector (New)
        self.orientationSelector = qt.QComboBox()
        self.orientationSelector.addItems(["Axial (Red)", "Coronal (Green)", "Sagittal (Yellow)"])
        self.orientationSelector.setToolTip("Select the slice orientation where Z-frame dots are visible.")
        parametersFormLayout.addRow("Detection Orientation: ", self.orientationSelector)
        self.orientationSelector.connect("currentIndexChanged(int)", self.onOrientationChanged)

        # Fiducial type selector
        self.fiducialTypeSelector = qt.QComboBox()
        self.fiducialTypeSelector.addItems(["7-fiducial", "9-fiducial"])
        parametersFormLayout.addRow("Fiducial Frame Type: ", self.fiducialTypeSelector)
        self.fiducialTypeSelector.setCurrentIndex(0)
        
        # Z-frame configuration selector
        self.zframeConfigSelector = qt.QComboBox()
        self.loadZFrameConfigs()
        parametersFormLayout.addRow("Z-Frame Configuration: ", self.zframeConfigSelector)
        self.zframeConfigSelector.setCurrentIndex(3)
        self.zframeConfigSelector.connect("currentTextChanged(QString)", self.onZFrameConfigChanged)

        # Frame Topology Text Edit
        self.frameTopologyTextEdit = qt.QTextEdit()
        self.frameTopologyTextEdit.setReadOnly(False)
        self.frameTopologyTextEdit.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.frameTopologyTextEdit.setMaximumHeight(40)
        parametersFormLayout.addRow("Frame Topology: ", self.frameTopologyTextEdit)
        self.onZFrameConfigChanged(self.zframeConfigSelector.currentText)

        # Slice range
        self.sliceRangeWidget = slicer.qMRMLRangeWidget()
        self.sliceRangeWidget.decimals = 0
        self.sliceRangeWidget.minimum = 0
        self.sliceRangeWidget.maximum = 100
        self.sliceRangeWidget.singleStep = 1
        parametersFormLayout.addRow("Slice Range: ", self.sliceRangeWidget)

        # Visualization Step (for decimation)
        self.visStepSpinBox = qt.QSpinBox()
        self.visStepSpinBox.setRange(1, 100)
        self.visStepSpinBox.setValue(1)
        self.visStepSpinBox.setToolTip("Reduce clutter by skipping slices during visualization.")
        parametersFormLayout.addRow("Visualization Step: ", self.visStepSpinBox)

        # Marker Diameter Setting
        self.markerDiameterSpinBox = qt.QSpinBox()
        self.markerDiameterSpinBox.setRange(3, 100)
        self.markerDiameterSpinBox.setValue(11) 
        self.markerDiameterSpinBox.setToolTip("Diameter of the fiducial marker in pixels.")
        parametersFormLayout.addRow("Marker Diameter (px): ", self.markerDiameterSpinBox)

        # Initialize UI state
        self.onInputVolumeSelected(self.inputSelector.currentNode())
        
        # Output transform selector
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.showHidden = False
        self.outputSelector.showChildNodeTypes = False
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("Pick the output transform")
        parametersFormLayout.addRow("Output Transform: ", self.outputSelector)
        
        # --- Actions Area ---
        actionsLayout = qt.QVBoxLayout()
        parametersFormLayout.addRow(actionsLayout)
        
        baseButtonStyle = "font-weight: bold; font-size: 13px; padding: 6px;"

        self.applyButton = qt.QPushButton("Run Registration")
        self.applyButton.setToolTip("Run registration (Compute transform).")
        self.applyButton.setStyleSheet(baseButtonStyle)
        self.applyButton.enabled = True
        actionsLayout.addWidget(self.applyButton)
        
        self.applyVisButton = qt.QPushButton("Visualize Detected Points")
        self.applyVisButton.setToolTip("Detect and visualize points ONLY (Does not update transform).")
        self.applyVisButton.setStyleSheet(baseButtonStyle + " color: #00aa00;")
        self.applyVisButton.enabled = True
        actionsLayout.addWidget(self.applyVisButton)
        
        self.visualizeButton = qt.QPushButton("Visualize Frame Topology")
        self.visualizeButton.toolTip = "Visualize the Z-frame topology definition."
        self.visualizeButton.setStyleSheet(baseButtonStyle + " color: #d60000;")
        self.visualizeButton.enabled = True
        actionsLayout.addWidget(self.visualizeButton)

        # Connect Buttons
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.applyVisButton.connect('clicked(bool)', self.onApplyVisButton)
        self.visualizeButton.connect('clicked(bool)', self.onVisualizeButton)
        
        self.layout.addStretch(1)

    def onVisualizeButton(self):
        try:
            self.logic.visualize_topology(self.frameTopologyTextEdit.toPlainText())
        except Exception as e:
            slicer.util.errorDisplay("Failed to visualize topology: "+str(e))
            import traceback
            traceback.print_exc()

    def onInputVolumeSelected(self, node):
        self.onOrientationChanged(self.orientationSelector.currentIndex)

    def onOrientationChanged(self, index):
        # Update slice range based on selected orientation
        node = self.inputSelector.currentNode()
        if node:
            dims = node.GetImageData().GetDimensions()
            # dims are (I, J, K)
            
            orientation = self.orientationSelector.currentText
            max_slice = 0
            
            if "Axial" in orientation:
                max_slice = dims[2] # K
            elif "Coronal" in orientation:
                max_slice = dims[1] # J
            elif "Sagittal" in orientation:
                max_slice = dims[0] # I
                
            self.sliceRangeWidget.maximum = max_slice
            self.sliceRangeWidget.minimum = 0
            self.sliceRangeWidget.minimumValue = 0
            self.sliceRangeWidget.maximumValue = max_slice

    def loadZFrameConfigs(self):
        configPath = os.path.join(os.path.dirname(__file__), 'Resources', 'configs.txt')
        self.zframeConfigSelector.clear()
        try:
            with open(configPath, 'r') as f:
                lines = f.readlines()
            self.zFrameTopologies = {}
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'): continue
                try:
                    config_name, topology = line.split(':', 1)
                    self.zFrameTopologies[config_name.strip()] = topology.strip()
                except ValueError: continue
            if self.zFrameTopologies:
                self.zframeConfigSelector.addItems(sorted(self.zFrameTopologies.keys()))
            else:
                self.zframeConfigSelector.addItems(["Configs not found"])
        except Exception as e:
            logging.error(f"Error loading Z-frame configurations: {str(e)}")
            self.zframeConfigSelector.addItems(["Configs not found"])

    def onZFrameConfigChanged(self, configName):
        self.frameTopologyTextEdit.setText(self.zFrameTopologies.get(configName, "Unknown configuration"))

    def onApplyButton(self):
        self.runRegistration(visualize=False, updateTransform=True)

    def onApplyVisButton(self):
        self.runRegistration(visualize=True, updateTransform=False)

    def runRegistration(self, visualize, updateTransform=True):
        try:
            self.logic.run(self.inputSelector.currentNode(),
                     self.outputSelector.currentNode(),
                     self.zframeConfigSelector.currentText,
                     self.fiducialTypeSelector.currentText,
                     self.frameTopologyTextEdit.toPlainText(),
                     int(self.sliceRangeWidget.minimumValue),
                     int(self.sliceRangeWidget.maximumValue),
                     visualize=visualize,
                     visualizeStep=int(self.visStepSpinBox.value),
                     markerDiameter=int(self.markerDiameterSpinBox.value),
                     updateTransform=updateTransform,
                     orientation=self.orientationSelector.currentText)
        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: "+str(e))
            import traceback
            traceback.print_exc()

class ZFrameRegistrationScriptedLogic(ScriptedLoadableModuleLogic):
    def run(self, inputVolume, outputTransform, zframeConfig, zframeType, frameTopology, startSlice, endSlice, visualize=False, visualizeStep=1, markerDiameter=11, updateTransform=True, orientation="Axial"):
        logging.info('Processing started')
        
        if not inputVolume or not outputTransform:
            raise ValueError("Input volume or output transform is missing")
            
        imageData = inputVolume.GetImageData()
        if not imageData:
            raise ValueError("Input image is invalid")
        
        dim = imageData.GetDimensions()
        # vtk_to_numpy returns flat array. reshape to (K, J, I).
        # Note: numpy shape is (z, y, x) corresponding to (K, J, I)
        imageDataArr = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
        imageDataArr = imageDataArr.reshape(dim[2], dim[1], dim[0]) 
        
        # Original code for Axial: (K, J, I) -> transpose(2, 1, 0) -> (I, J, K)
        # We need the last dimension to be the slicing direction.
        # Input to ZFrameRegistration is expected to be [ImageX, ImageY, SliceIndex]
        
        origin = inputVolume.GetOrigin()
        spacing = inputVolume.GetSpacing()
        directions = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASDirectionMatrix(directions)

        # Base imageTransform construction (I, J, K columns)
        imageTransform = np.eye(4)
        for i in range(3):
            for j in range(3):
                imageTransform[i,j] = spacing[j] * directions.GetElement(i,j)
            imageTransform[i,3] = origin[i]

        # --- Handle Orientation Swapping ---
        # 1. Permute imageDataArr so dim 2 is the slice axis.
        # 2. Permute imageTransform columns so they match the new (ImageX, ImageY, Slice) axes.
        
        if "Axial" in orientation:
            # (K, J, I) -> (I, J, K)
            # ImageX=I, ImageY=J, Slice=K
            imageDataArr = imageDataArr.transpose(2,1,0)
            # Matrix columns: 0=I, 1=J, 2=K (No change)
            
        elif "Coronal" in orientation:
            # Slicer Green: Slice along J axis.
            # Slice Plane is I-K (X-Z). Usually ImageX=I, ImageY=K.
            # Original Array: (K, J, I)
            # Target: (I, K, J)
            # Transpose: (2, 0, 1) relative to original (K, J, I) 
            # Wait: dim 2 is I, dim 0 is K, dim 1 is J.
            imageDataArr = imageDataArr.transpose(2, 0, 1)
            
            # Matrix Permutation:
            # New Col 0 (ImageX) <- Old Col 0 (I)
            # New Col 1 (ImageY) <- Old Col 2 (K)
            # New Col 2 (Slice)  <- Old Col 1 (J)
            original_matrix = imageTransform.copy()
            imageTransform[:, 0] = original_matrix[:, 0] # I
            imageTransform[:, 1] = original_matrix[:, 2] # K
            imageTransform[:, 2] = original_matrix[:, 1] # J

        elif "Sagittal" in orientation:
            # Slicer Yellow: Slice along I axis.
            # Slice Plane is J-K (Y-Z). Usually ImageX=J, ImageY=K.
            # Original Array: (K, J, I)
            # Target: (J, K, I)
            # Transpose: (1, 0, 2) relative to original (K, J, I)
            # dim 1 is J, dim 0 is K, dim 2 is I.
            imageDataArr = imageDataArr.transpose(1, 0, 2)
            
            # Matrix Permutation:
            # New Col 0 (ImageX) <- Old Col 1 (J)
            # New Col 1 (ImageY) <- Old Col 2 (K)
            # New Col 2 (Slice)  <- Old Col 0 (I)
            original_matrix = imageTransform.copy()
            imageTransform[:, 0] = original_matrix[:, 1] # J
            imageTransform[:, 1] = original_matrix[:, 2] # K
            imageTransform[:, 2] = original_matrix[:, 0] # I

        # -----------------------------------

        ZmatrixBase = np.eye(4)
        ZquaternionBase = zf.MatrixToQuaternion(ZmatrixBase)

        sliceRange = [startSlice, endSlice]
        
        frameTopologyArr = []
        try:
            frameTopologyList = ''.join(frameTopology.split()).strip("[]").split("],[")
            for n in frameTopologyList:
                x, y, z = map(float, n.split(","))
                frameTopologyArr.append([x, y, z])
        except Exception as e:
             raise ValueError(f"Error parsing topology: {e}")

        if zframeType == "7-fiducial":
            registration = ZFrameRegistration(numFiducials=7)
        elif zframeType == "9-fiducial":
            registration = ZFrameRegistration(numFiducials=9)
        else:
            raise ValueError("Invalid Z-frame configuration")
        
        result = False
        all_detected_points = []

        if registration:
            registration.SetMarkerDiameter(markerDiameter)
            registration.SetInputImage(imageDataArr, imageTransform)
            registration.SetOrientationBase(ZquaternionBase)
            registration.SetFrameTopology(frameTopologyArr)
            
            result, Zposition, Zorientation, all_detected_points = registration.Register(sliceRange)
        
        # --- Visualization of detected points ---
        if visualize and all_detected_points:
            points_to_visualize = all_detected_points[::visualizeStep]
            print(f"Visualizing points for {len(points_to_visualize)} slices (Step: {visualizeStep})")
            # Pass orientation to visualize function to map coordinates back correctly
            self.visualize_detected_points(inputVolume, points_to_visualize, orientation)
        # -----------------------

        if result and updateTransform:
            matrix = zf.QuaternionToMatrix(Zorientation)
            zMatrix = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    zMatrix.SetElement(i,j, matrix[i][j])
                zMatrix.SetElement(i,3, Zposition[i])
            
            outputTransform.SetMatrixTransformToParent(zMatrix)
            logging.info('Processing completed')
            return True
        elif result and not updateTransform:
            logging.info('Detection completed (Transform update skipped)')
            return True
        else:
            logging.error('Processing failed')
            return False

    def clear_detected_points_visualization(self):
        nodeName = "Detected Z-Frame Points"
        markupsNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if markupsNode:
            markupsNode.RemoveAllControlPoints()

    def visualize_detected_points(self, inputVolume, points_list, orientation="Axial"):
        """
        points_list: [{"slice": int, "points": [[x,y],...]}, ...]
        orientation: "Axial", "Coronal", or "Sagittal"
        """
        nodeName = "Detected Z-Frame Points"
        markupsNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if not markupsNode:
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", nodeName)
        
        markupsNode.RemoveAllControlPoints()
        markupsNode.GetDisplayNode().SetSelectedColor(0, 1, 0) # Green
        
        markupsNode.GetDisplayNode().SetTextScale(3.0) 
        markupsNode.GetDisplayNode().SetGlyphScale(3.0)
        
        # IJK to RAS Matrix (This expects original IJK)
        ijkToRas = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRas)
        
        for item in points_list:
            slice_idx = item["slice"]
            points = item["points"]
            
            for i, pt in enumerate(points):
                # Map algorithm coordinates back to Volume IJK based on orientation
                if "Axial" in orientation:
                    # Alg: X=I, Y=J, Z=K
                    ijk = [pt[0], pt[1], slice_idx, 1.0]
                elif "Coronal" in orientation:
                    # Alg: X=I, Y=K, Z=J
                    # Volume IJK: I=pt[0], J=slice_idx, K=pt[1]
                    ijk = [pt[0], slice_idx, pt[1], 1.0]
                elif "Sagittal" in orientation:
                    # Alg: X=J, Y=K, Z=I
                    # Volume IJK: I=slice_idx, J=pt[0], K=pt[1]
                    ijk = [slice_idx, pt[0], pt[1], 1.0]
                
                ras = [0.0]*4
                ijkToRas.MultiplyPoint(ijk, ras)
                
                label = f"{slice_idx}-{i}"
                markupsNode.AddControlPoint(vtk.vtkVector3d(ras[0], ras[1], ras[2]), label)
            
        if points_list:
             # Jump to the last detected point
             last_item = points_list[-1]
             last_slice = last_item["slice"]
             last_pt = last_item["points"][0]
             
             if "Axial" in orientation:
                 ijk = [last_pt[0], last_pt[1], last_slice, 1.0]
             elif "Coronal" in orientation:
                 ijk = [last_pt[0], last_slice, last_pt[1], 1.0]
             elif "Sagittal" in orientation:
                 ijk = [last_slice, last_pt[0], last_pt[1], 1.0]
                 
             ras = [0.0]*4
             ijkToRas.MultiplyPoint(ijk, ras)
             slicer.modules.markups.logic().JumpSlicesToLocation(ras[0], ras[1], ras[2], True)

    def visualize_topology(self, frameTopology):
        logging.info('Visualizing Z-frame topology with Lines and Labeled Points.')
        frameTopologyArr = []
        try:
            # Parse topology string
            frameTopologyList = ''.join(frameTopology.split()).strip("[]").split("],[")
            for n in frameTopologyList:
                x, y, z = map(float, n.split(","))
                frameTopologyArr.append([x, y, z])
        except ValueError:
            slicer.util.errorDisplay("Topology format error.")
            return

        # Get origins and vectors
        origins = frameTopologyArr[0:3]
        vectors = frameTopologyArr[3:6]
        
        # 1. Model node for drawing lines
        vtk_points_lines = vtk.vtkPoints()
        vtk_lines = vtk.vtkCellArray()
        
        pid = 0
        for o, v in zip(origins, vectors):
            start_pt = np.array(o)
            end_pt = np.array(o) + np.array(v)
            
            # Add line coordinates
            vtk_points_lines.InsertNextPoint(start_pt)
            vtk_points_lines.InsertNextPoint(end_pt)
            
            # Create line
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid)
            line.GetPointIds().SetId(1, pid + 1)
            vtk_lines.InsertNextCell(line)
            pid += 2

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(vtk_points_lines)
        polyData.SetLines(vtk_lines)

        lineNodeName = "Z-Frame Topology Lines"
        lineModelNode = slicer.mrmlScene.GetFirstNodeByName(lineNodeName)
        if not lineModelNode:
            lineModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", lineNodeName)
            lineModelNode.CreateDefaultDisplayNodes()
            
        lineModelNode.SetAndObservePolyData(polyData)
        
        # Set line color to "Red"
        lineDisplayNode = lineModelNode.GetDisplayNode()
        if lineDisplayNode:
            lineDisplayNode.SetColor(1, 0, 0)
            lineDisplayNode.SetLineWidth(4.0)
            lineDisplayNode.SetOpacity(1.0)

        # 2. Markup node for displaying named points
        pointsNodeName = "Z-Frame Topology Points"
        markupsNode = slicer.mrmlScene.GetFirstNodeByName(pointsNodeName)
        if not markupsNode:
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", pointsNodeName)
        
        markupsNode.RemoveAllControlPoints()
        markupsNode.GetDisplayNode().SetSelectedColor(1, 0, 0)
        markupsNode.GetDisplayNode().SetColor(1, 0, 0)
        markupsNode.GetDisplayNode().SetTextScale(4.0)
        markupsNode.GetDisplayNode().SetGlyphScale(3.0)
        
        for i, (o, v) in enumerate(zip(origins, vectors)):
            start_pt = np.array(o)
            end_pt = np.array(o) + np.array(v)
            markupsNode.AddControlPoint(vtk.vtkVector3d(start_pt), f"Rod{i+1}-Start")
            markupsNode.AddControlPoint(vtk.vtkVector3d(end_pt), f"Rod{i+1}-End")

        print(f"Created topology visualization: Lines and Labeled Points (Red)")