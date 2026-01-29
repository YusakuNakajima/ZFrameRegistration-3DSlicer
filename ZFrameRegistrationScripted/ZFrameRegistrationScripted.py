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
        self.markerDiameterSpinBox.setValue(5)
        self.markerDiameterSpinBox.setToolTip("Diameter of the fiducial marker in pixels.")
        parametersFormLayout.addRow("Marker Diameter (px): ", self.markerDiameterSpinBox)

        # Diagnostic Logging Controls
        self.enableDiagnosticsCheckbox = qt.QCheckBox()
        self.enableDiagnosticsCheckbox.setChecked(True)
        self.enableDiagnosticsCheckbox.setToolTip("Enable JSONL diagnostic logging for registration analysis.")
        parametersFormLayout.addRow("Enable Diagnostic Logging: ", self.enableDiagnosticsCheckbox)

        self.diagnosticsPathLineEdit = qt.QLineEdit()
        self.diagnosticsPathLineEdit.setPlaceholderText("Leave empty for default: ~/zframe_diagnostics/")
        self.diagnosticsPathLineEdit.setToolTip("Optional custom path for diagnostic JSONL output file.")
        self.diagnosticsPathButton = qt.QPushButton("Browse...")
        self.diagnosticsPathButton.setMaximumWidth(80)
        diagnosticsPathLayout = qt.QHBoxLayout()
        diagnosticsPathLayout.addWidget(self.diagnosticsPathLineEdit)
        diagnosticsPathLayout.addWidget(self.diagnosticsPathButton)
        parametersFormLayout.addRow("Diagnostic Output Path: ", diagnosticsPathLayout)
        self.diagnosticsPathButton.connect('clicked(bool)', self.onDiagnosticsPathBrowse)

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

        # --- Auto Orient (Pre-processing) ---
        autoOrientCollapsibleButton = ctk.ctkCollapsibleButton()
        autoOrientCollapsibleButton.text = "Auto Orient (Pre-processing)"
        autoOrientCollapsibleButton.collapsed = True
        self.layout.addWidget(autoOrientCollapsibleButton)
        autoOrientFormLayout = qt.QFormLayout(autoOrientCollapsibleButton)

        # Input volume selector for Auto Orient (separate from main panel)
        self.autoOrientInputSelector = slicer.qMRMLNodeComboBox()
        self.autoOrientInputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.autoOrientInputSelector.selectNodeUponCreation = True
        self.autoOrientInputSelector.addEnabled = False
        self.autoOrientInputSelector.removeEnabled = False
        self.autoOrientInputSelector.noneEnabled = False
        self.autoOrientInputSelector.showHidden = False
        self.autoOrientInputSelector.showChildNodeTypes = False
        self.autoOrientInputSelector.setMRMLScene(slicer.mrmlScene)
        self.autoOrientInputSelector.setToolTip("Select the input volume for auto-orientation (typically a cropped Z-frame region).")
        autoOrientFormLayout.addRow("Input Volume: ", self.autoOrientInputSelector)

        # Orientation selector for Auto Orient (separate from main panel)
        self.autoOrientOrientationSelector = qt.QComboBox()
        self.autoOrientOrientationSelector.addItems(["Axial (Red)", "Coronal (Green)", "Sagittal (Yellow)"])
        self.autoOrientOrientationSelector.setToolTip("Select the target orientation axis for alignment.")
        autoOrientFormLayout.addRow("Target Orientation: ", self.autoOrientOrientationSelector)

        # Enable Roll Fix checkbox
        self.enableRollFixCheckbox = qt.QCheckBox()
        self.enableRollFixCheckbox.setChecked(True)  # Default: enabled
        autoOrientFormLayout.addRow("Enable Roll Fix (PC2): ", self.enableRollFixCheckbox)

        # Lambda2/Lambda3 threshold spin box
        self.lambdaRatioThresholdSpinBox = qt.QDoubleSpinBox()
        self.lambdaRatioThresholdSpinBox.setRange(1.0, 10.0)
        self.lambdaRatioThresholdSpinBox.setValue(1.5)
        self.lambdaRatioThresholdSpinBox.setSingleStep(0.1)
        autoOrientFormLayout.addRow("Lambda2/Lambda3 Threshold: ", self.lambdaRatioThresholdSpinBox)

        # Roll offset spin box (additional rotation after PC2-based roll fix)
        self.rollOffsetSpinBox = qt.QDoubleSpinBox()
        self.rollOffsetSpinBox.setRange(-180.0, 180.0)
        self.rollOffsetSpinBox.setValue(0.0)
        self.rollOffsetSpinBox.setSingleStep(15.0)
        self.rollOffsetSpinBox.setSuffix(" deg")
        self.rollOffsetSpinBox.setToolTip("Additional roll offset applied after PC2-based roll fix (degrees)")
        autoOrientFormLayout.addRow("Roll Offset: ", self.rollOffsetSpinBox)

        self.orientTransformSelector = slicer.qMRMLNodeComboBox()
        self.orientTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
        self.orientTransformSelector.selectNodeUponCreation = True
        self.orientTransformSelector.addEnabled = True
        self.orientTransformSelector.removeEnabled = True
        self.orientTransformSelector.noneEnabled = True
        self.orientTransformSelector.showHidden = False
        self.orientTransformSelector.showChildNodeTypes = False
        self.orientTransformSelector.setMRMLScene(slicer.mrmlScene)
        self.orientTransformSelector.setToolTip("Output transform for auto-orientation.")
        autoOrientFormLayout.addRow("Output Transform: ", self.orientTransformSelector)

        self.autoOrientButton = qt.QPushButton("Auto Orient")
        self.autoOrientButton.setToolTip("Detect Z-frame rod principal axis via PCA and align to the selected Target Orientation axis.")
        self.autoOrientButton.setStyleSheet(baseButtonStyle + " color: #0055aa;")
        self.autoOrientButton.enabled = True
        autoOrientFormLayout.addRow(self.autoOrientButton)

        self.autoOrientStatusLabel = qt.QLabel("")
        autoOrientFormLayout.addRow("Status: ", self.autoOrientStatusLabel)

        self.autoOrientButton.connect('clicked(bool)', self.onAutoOrientButton)

        self.layout.addStretch(1)

    def onAutoOrientButton(self):
        inputVolume = self.autoOrientInputSelector.currentNode()
        orientTransform = self.orientTransformSelector.currentNode()
        if not inputVolume:
            self.autoOrientStatusLabel.setText("FAILED: No input volume selected")
            return
        if not orientTransform:
            self.autoOrientStatusLabel.setText("FAILED: No output transform selected")
            return
        try:
            orientation = self.autoOrientOrientationSelector.currentText
            enableRollFix = self.enableRollFixCheckbox.isChecked()
            lambdaRatioThreshold = self.lambdaRatioThresholdSpinBox.value
            rollOffsetDeg = self.rollOffsetSpinBox.value
            success, message = self.logic.autoOrient(inputVolume, orientTransform, orientation,
                                                      enableRollFix=enableRollFix,
                                                      lambdaRatioThreshold=lambdaRatioThreshold,
                                                      rollOffsetDeg=rollOffsetDeg)
            if success:
                self.autoOrientStatusLabel.setText("OK: " + message)
            else:
                self.autoOrientStatusLabel.setText("SKIPPED: " + message)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.autoOrientStatusLabel.setText("FAILED: " + str(e))

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

    def onDiagnosticsPathBrowse(self):
        path = qt.QFileDialog.getSaveFileName(
            self.parent, "Select Diagnostic Output File",
            "", "JSONL Files (*.jsonl);;All Files (*)"
        )
        if path:
            self.diagnosticsPathLineEdit.setText(path)

    def onApplyButton(self):
        self.runRegistration(visualize=False, updateTransform=True)

    def onApplyVisButton(self):
        self.runRegistration(visualize=True, updateTransform=False)

    def runRegistration(self, visualize, updateTransform=True):
        try:
            enableDiagnostics = self.enableDiagnosticsCheckbox.isChecked()
            diagnosticsPath = self.diagnosticsPathLineEdit.text.strip() if self.diagnosticsPathLineEdit.text.strip() else None
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
                     orientation=self.orientationSelector.currentText,
                     enableDiagnostics=enableDiagnostics,
                     diagnosticsPath=diagnosticsPath)
        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: "+str(e))
            import traceback
            traceback.print_exc()

class ZFrameRegistrationScriptedLogic(ScriptedLoadableModuleLogic):
    def run(self, inputVolume, outputTransform, zframeConfig, zframeType, frameTopology, startSlice, endSlice, visualize=False, visualizeStep=1, markerDiameter=11, updateTransform=True, orientation="Axial", enableDiagnostics=False, diagnosticsPath=None):
        logging.info('Processing started')

        if not inputVolume or not outputTransform:
            raise ValueError("Input volume or output transform is missing")

        # Initialize diagnostic logger if enabled
        logger = None
        if enableDiagnostics:
            from ZFrame.DiagnosticLogger import ZFrameDiagnosticLogger
            logger = ZFrameDiagnosticLogger(output_path=diagnosticsPath, enabled=True)
            
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
            registration = ZFrameRegistration(numFiducials=7, logger=logger)
        elif zframeType == "9-fiducial":
            registration = ZFrameRegistration(numFiducials=9, logger=logger)
        else:
            raise ValueError("Invalid Z-frame configuration")

        result = False
        all_detected_points = []

        if registration:
            registration.SetMarkerDiameter(markerDiameter)
            registration.SetInputImage(imageDataArr, imageTransform)
            registration.SetOrientationBase(ZquaternionBase)
            registration.SetFrameTopology(frameTopologyArr)

            # Start logger session if enabled
            if logger:
                logger.start_session({
                    'num_fiducials': 7 if zframeType == "7-fiducial" else 9,
                    'marker_diameter': markerDiameter,
                    'orientation': orientation,
                    'frame_config': zframeConfig,
                    'slice_range': [startSlice, endSlice],
                    'frame_topology': frameTopologyArr
                })

            result, Zposition, Zorientation, all_detected_points, rms_error = registration.Register(sliceRange, logger=logger)
        
        # --- Visualization of detected points ---
        if visualize and all_detected_points:
            points_to_visualize = all_detected_points[::visualizeStep]
            print(f"Visualizing points for {len(points_to_visualize)} slices (Step: {visualizeStep})")
            # Pass orientation to visualize function to map coordinates back correctly
            self.visualize_detected_points(inputVolume, points_to_visualize, orientation)
        # -----------------------

        # --- Print detected / template points in RAS (mm) ---
        if result and all_detected_points:
            ijkToRas = vtk.vtkMatrix4x4()
            inputVolume.GetIJKToRASMatrix(ijkToRas)

            print("\n=== Detected Points (RAS mm) ===")
            for item in all_detected_points:
                slice_idx = item["slice"]
                points = item["points"]
                for i, pt in enumerate(points):
                    if "Axial" in orientation:
                        ijk = [pt[0], pt[1], slice_idx, 1.0]
                    elif "Coronal" in orientation:
                        ijk = [pt[0], slice_idx, pt[1], 1.0]
                    elif "Sagittal" in orientation:
                        ijk = [slice_idx, pt[0], pt[1], 1.0]
                    ras = [0.0] * 4
                    ijkToRas.MultiplyPoint(ijk, ras)
                    print(f"  Slice {slice_idx}, Fid {i}: RAS = ({ras[0]:.2f}, {ras[1]:.2f}, {ras[2]:.2f})")

            regMatrix = np.eye(4)
            regMatrix[:3, :3] = zf.QuaternionToMatrix(Zorientation)[:3, :3]
            regMatrix[0:3, 3] = Zposition

            print("\n=== Template Points (RAS mm) ===")
            origins = frameTopologyArr[0:3]
            vectors = frameTopologyArr[3:6]
            for i, (o, v) in enumerate(zip(origins, vectors)):
                start_frame = np.array([o[0], o[1], o[2], 1.0])
                end_frame = np.array([o[0] + v[0], o[1] + v[1], o[2] + v[2], 1.0])
                start_ras = regMatrix @ start_frame
                end_ras = regMatrix @ end_frame
                print(f"  Rod {i+1} Start: RAS = ({start_ras[0]:.2f}, {start_ras[1]:.2f}, {start_ras[2]:.2f})")
                print(f"  Rod {i+1} End:   RAS = ({end_ras[0]:.2f}, {end_ras[1]:.2f}, {end_ras[2]:.2f})")
        # -------------------------------------------------

        if result and updateTransform:
            matrix = zf.QuaternionToMatrix(Zorientation)
            zMatrix = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    zMatrix.SetElement(i,j, matrix[i][j])
                zMatrix.SetElement(i,3, Zposition[i])

            outputTransform.SetMatrixTransformToParent(zMatrix)

            # --- Rotation sanity check ---
            R = np.array([[zMatrix.GetElement(i, j) for j in range(3)] for i in range(3)])
            det_R = np.linalg.det(R)
            orth_err = np.linalg.norm(R.T @ R - np.eye(3))
            print(f"\n=== Output Transform Rotation Check ===")
            print(f"  det(R)    = {det_R:.6f}   (expect +1)")
            print(f"  orth_err  = {orth_err:.6e}   (expect ~0)")
            if abs(det_R - 1.0) > 0.01 or orth_err > 0.01:
                print("  WARNING: rotation matrix may be invalid (det != +1 or poor orthogonality)")
            else:
                print("  OK: valid rotation matrix")
            # -----------------------------

            # End diagnostic logging session
            if logger:
                logger.end_session()
                diag_path = str(logger.output_path)
                logging.info(f"Diagnostics written to: {diag_path}")
                print(f"\n=== Diagnostic Logging ===")
                print(f"  Output file: {diag_path}")

            logging.info('Processing completed')
            if rms_error is not None:
                rms_msg = f"RMS Error: {rms_error:.4f} mm"
                logging.info(rms_msg)
                slicer.util.infoDisplay(rms_msg, windowTitle="Registration Result")
            return True, rms_error
        elif result and not updateTransform:
            # End diagnostic logging session
            if logger:
                logger.end_session()
                diag_path = str(logger.output_path)
                logging.info(f"Diagnostics written to: {diag_path}")
                print(f"\n=== Diagnostic Logging ===")
                print(f"  Output file: {diag_path}")

            logging.info('Detection completed (Transform update skipped)')
            if rms_error is not None:
                rms_msg = f"RMS Error: {rms_error:.4f} mm"
                logging.info(rms_msg)
                slicer.util.infoDisplay(rms_msg, windowTitle="Detection Result")
            return True, rms_error
        else:
            # End diagnostic logging session even on failure
            if logger:
                logger.end_session()
                diag_path = str(logger.output_path)
                logging.info(f"Diagnostics written to: {diag_path}")
                print(f"\n=== Diagnostic Logging ===")
                print(f"  Output file: {diag_path}")

            logging.error('Processing failed')
            slicer.util.errorDisplay("Z-Frame registration failed. No valid slices found.\nPlease check:\n1. 'Scripted Registration Parameters' are used.\n2. Marker Diameter is correct.\n3. Slice Range covers the frame.\n4. View 'Python Interactor' for debug logs.")
            return False, None

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

    # ------------------------------------------------------------------
    # Auto Orient helpers
    # ------------------------------------------------------------------

    def autoOrient(self, inputVolume, outputTransform, orientation,
                   enableRollFix=True, lambdaRatioThreshold=1.5, rollOffsetDeg=0.0):
        """Align the Z-frame principal axis to the selected orientation axis.

        Args:
            inputVolume: Input volume node
            outputTransform: Output transform node
            orientation: Target orientation ("Axial", "Coronal", or "Sagittal")
            enableRollFix: If True, apply PC2-based roll fix (Step B)
            lambdaRatioThreshold: Minimum lambda2/lambda3 ratio to apply roll fix
            rollOffsetDeg: Additional roll offset in degrees (applied after PC2 roll fix)

        Returns:
            (bool, str): (True, message) on success, (False, reason) on skip/failure.
        """
        logging.info("AutoOrient: starting")

        imageData = inputVolume.GetImageData()
        if not imageData:
            return (False, "Input image is invalid")

        # Numpy array (flat -> 3-D)
        dims = imageData.GetDimensions()  # (I, J, K)
        arr = vtk.util.numpy_support.vtk_to_numpy(
            imageData.GetPointData().GetScalars()
        ).reshape(dims[2], dims[1], dims[0])  # (K, J, I)

        # Otsu threshold
        threshold = self._otsuThreshold(arr)
        bright_mask = arr > threshold
        bright_count = int(np.count_nonzero(bright_mask))
        logging.info(f"AutoOrient: Otsu threshold={threshold}, bright voxels={bright_count}")

        if bright_count < 50:
            return (False, "Too few bright voxels")

        # IJK coordinates of bright voxels
        kk, jj, ii = np.nonzero(bright_mask)
        ijk_coords = np.column_stack([ii, jj, kk]).astype(np.float64)  # (N, 3)

        # IJK -> RAS
        ijkToRas = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRas)
        ijkToRas_np = np.eye(4)
        for r in range(4):
            for c in range(4):
                ijkToRas_np[r, c] = ijkToRas.GetElement(r, c)

        ones = np.ones((ijk_coords.shape[0], 1))
        ijk_h = np.hstack([ijk_coords, ones])  # (N, 4)
        ras_coords = (ijkToRas_np @ ijk_h.T).T[:, :3]  # (N, 3)

        # PCA via covariance eigen-decomposition
        center = ras_coords.mean(axis=0)
        centered = ras_coords - center
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns ascending order: [lambda3, lambda2, lambda1] (smallest to largest)
        lambda3, lambda2, lambda1 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
        pc3, pc2, pc1 = eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2]

        # Normalize principal axes
        pc1 = pc1 / np.linalg.norm(pc1)
        pc2 = pc2 / np.linalg.norm(pc2)

        logging.info(f"AutoOrient: PC1 (principal axis) = {pc1}")
        logging.info(f"AutoOrient: PC2 (secondary axis) = {pc2}")
        logging.info(f"AutoOrient: eigenvalues = lambda1={lambda1:.4f}, lambda2={lambda2:.4f}, lambda3={lambda3:.4f}")

        if lambda2 > 1e-12:
            elongation = lambda1 / lambda2
            logging.info(f"AutoOrient: elongation ratio (lambda1/lambda2) = {elongation:.2f}")

        # Target axis from orientation
        if "Axial" in orientation:
            target_axis = np.array([0.0, 0.0, 1.0])
        elif "Coronal" in orientation:
            target_axis = np.array([0.0, 1.0, 0.0])
        elif "Sagittal" in orientation:
            target_axis = np.array([1.0, 0.0, 0.0])
        else:
            target_axis = np.array([0.0, 0.0, 1.0])

        dot_val = float(np.dot(pc1, target_axis))
        logging.info(f"AutoOrient: dot(PC1, target) = {dot_val:.6f}")

        # ===== Step A: PC1 alignment (existing logic) =====
        if abs(dot_val) > 0.98:
            # Already aligned – use identity for R1
            R1 = np.eye(3)
            logging.info("AutoOrient: PC1 already well-aligned, R1 = identity")
        else:
            # Compute R1: rotate PC1 to target_axis
            R1 = self._computeAlignmentRotation3x3(pc1, target_axis)
            logging.info(f"AutoOrient: R1 computed (PC1 -> target_axis)")

        # ===== Step B: Roll fix using PC2 =====
        R_final = R1  # Default: no roll fix
        roll_applied = False
        roll_angle_deg = 0.0

        if enableRollFix:
            # Check lambda2/lambda3 ratio
            if lambda3 > 1e-12:
                lambda_ratio = lambda2 / lambda3
            else:
                lambda_ratio = float('inf')

            logging.info(f"AutoOrient: lambda2/lambda3 ratio = {lambda_ratio:.4f}, threshold = {lambdaRatioThreshold}")

            if lambda_ratio >= lambdaRatioThreshold:
                # Rotate PC2 by R1
                pc2_rotated = R1 @ pc2
                logging.info(f"AutoOrient: PC2 after R1 rotation = {pc2_rotated}")

                # Compute R2 for roll fix
                R2, roll_angle_deg, t2 = self._computeRollFixRotation(pc2_rotated, target_axis, orientation)

                logging.info(f"AutoOrient: Template secondary axis (t2) = {t2}")
                logging.info(f"AutoOrient: Roll fix angle = {roll_angle_deg:.2f} degrees")

                # Final rotation: R = R2 @ R1
                R_final = R2 @ R1
                roll_applied = True
                logging.info("AutoOrient: Roll fix applied (R = R2 @ R1)")
            else:
                logging.info(f"AutoOrient: Roll fix SKIPPED (lambda ratio {lambda_ratio:.4f} < threshold {lambdaRatioThreshold})")
        else:
            logging.info("AutoOrient: Roll fix disabled by user")

        # ===== Step C: Apply roll offset if specified =====
        if abs(rollOffsetDeg) > 0.01:
            R_offset = self._computeAxisRotation(target_axis, rollOffsetDeg)
            R_final = R_offset @ R_final
            logging.info(f"AutoOrient: Roll offset applied = {rollOffsetDeg:.1f} degrees")
        else:
            logging.info("AutoOrient: No roll offset applied")

        # Build 4x4 transformation matrix (rotate about center)
        c = np.asarray(center, dtype=np.float64)
        mat4 = np.eye(4)
        mat4[:3, :3] = R_final
        mat4[:3, 3] = c - R_final @ c

        # Debug: print final rotation matrix
        logging.info(f"AutoOrient: Final rotation matrix R:\n{R_final}")

        vtkMat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtkMat.SetElement(r, c, mat4[r, c])
        outputTransform.SetMatrixTransformToParent(vtkMat)

        logging.info("AutoOrient: transform set successfully")

        # Build status message
        parts = ["Orientation applied"]
        if roll_applied:
            parts.append(f"roll={roll_angle_deg:.1f}deg")
        if abs(rollOffsetDeg) > 0.01:
            parts.append(f"offset={rollOffsetDeg:.1f}deg")
        return (True, ", ".join(parts))

    @staticmethod
    def _otsuThreshold(arr):
        """Compute Otsu threshold using a 256-bin histogram (numpy only)."""
        arr_flat = arr.ravel().astype(np.float64)
        min_val, max_val = float(arr_flat.min()), float(arr_flat.max())
        if max_val == min_val:
            return min_val

        num_bins = 256
        hist, bin_edges = np.histogram(arr_flat, bins=num_bins, range=(min_val, max_val))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hist = hist.astype(np.float64)

        total = hist.sum()
        sum_total = (hist * bin_centers).sum()

        weight_bg = 0.0
        sum_bg = 0.0
        best_thresh = bin_centers[0]
        best_var = -1.0

        for i in range(num_bins):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break

            sum_bg += hist[i] * bin_centers[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg

            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thresh = bin_centers[i]

        return best_thresh

    @staticmethod
    def _computeAlignmentRotation(source_axis, target_axis, center_point):
        """Rodrigues rotation from *source_axis* to *target_axis* about *center_point*.

        Handles PCA sign ambiguity: if dot < 0, source is flipped before computing rotation.
        Returns a 4x4 homogeneous transformation matrix (numpy).
        """
        src = source_axis / np.linalg.norm(source_axis)
        tgt = target_axis / np.linalg.norm(target_axis)

        # Resolve sign ambiguity
        if np.dot(src, tgt) < 0:
            src = -src

        cross = np.cross(src, tgt)
        sin_a = np.linalg.norm(cross)
        cos_a = np.dot(src, tgt)

        if sin_a < 1e-10:
            # Vectors are (anti-)parallel – return identity
            T = np.eye(4)
            return T

        k = cross / sin_a  # unit rotation axis

        # Skew-symmetric matrix K
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

        # Build 4x4: rotate about center_point
        c = np.asarray(center_point, dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c - R @ c

        return T

    @staticmethod
    def _computeAlignmentRotation3x3(source_axis, target_axis):
        """Rodrigues rotation from *source_axis* to *target_axis*.

        Handles PCA sign ambiguity: if dot < 0, source is flipped before computing rotation.
        Returns a 3x3 rotation matrix (numpy) - for composition with other rotations.
        """
        src = source_axis / np.linalg.norm(source_axis)
        tgt = target_axis / np.linalg.norm(target_axis)

        # Resolve sign ambiguity
        if np.dot(src, tgt) < 0:
            src = -src

        cross = np.cross(src, tgt)
        sin_a = np.linalg.norm(cross)
        cos_a = np.dot(src, tgt)

        if sin_a < 1e-10:
            # Vectors are (anti-)parallel – return identity
            return np.eye(3)

        k = cross / sin_a  # unit rotation axis

        # Skew-symmetric matrix K
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

        return R

    @staticmethod
    def _computeRollFixRotation(pc2_rotated, target_axis, orientation):
        """Compute R2 rotation to fix roll using PC2.

        Args:
            pc2_rotated: PC2 after R1 rotation (3D vector)
            target_axis: The principal axis target (e.g., [0,0,1] for Axial)
            orientation: "Axial", "Coronal", or "Sagittal"

        Returns:
            (R2, angle_deg, t2): 3x3 rotation matrix, angle in degrees, template secondary axis
        """
        # Define template secondary axis t2 based on orientation
        # This is the expected direction of PC2 after proper alignment
        if "Axial" in orientation:
            # Z-frame rods along Z axis, secondary spread in X direction
            t2 = np.array([1.0, 0.0, 0.0])
        elif "Coronal" in orientation:
            # Z-frame rods along Y axis, secondary spread in X direction
            t2 = np.array([1.0, 0.0, 0.0])
        elif "Sagittal" in orientation:
            # Z-frame rods along X axis, secondary spread in Y direction
            t2 = np.array([0.0, 1.0, 0.0])
        else:
            t2 = np.array([1.0, 0.0, 0.0])

        # Project pc2_rotated onto the plane perpendicular to target_axis
        # p = v - (v . n) * n, where n is target_axis
        target_axis = target_axis / np.linalg.norm(target_axis)
        dot_proj = np.dot(pc2_rotated, target_axis)
        pc2_in_plane = pc2_rotated - dot_proj * target_axis

        # Normalize the projection
        pc2_in_plane_norm = np.linalg.norm(pc2_in_plane)
        if pc2_in_plane_norm < 1e-10:
            # PC2 is parallel to target axis - can't determine roll
            return np.eye(3), 0.0, t2

        pc2_in_plane = pc2_in_plane / pc2_in_plane_norm

        # t2 should already be in the perpendicular plane, but project just in case
        t2_in_plane = t2 - np.dot(t2, target_axis) * target_axis
        t2_in_plane_norm = np.linalg.norm(t2_in_plane)
        if t2_in_plane_norm < 1e-10:
            return np.eye(3), 0.0, t2
        t2_in_plane = t2_in_plane / t2_in_plane_norm

        # Compute signed angle between pc2_in_plane and t2_in_plane
        # Rotation axis is target_axis
        cos_angle = np.clip(np.dot(pc2_in_plane, t2_in_plane), -1.0, 1.0)
        cross_prod = np.cross(pc2_in_plane, t2_in_plane)
        sin_angle = np.dot(cross_prod, target_axis)  # Signed based on rotation axis

        angle_rad = np.arctan2(sin_angle, cos_angle)
        angle_deg = np.degrees(angle_rad)

        # Build rotation matrix R2: rotate about target_axis by angle_rad
        if abs(angle_rad) < 1e-10:
            return np.eye(3), 0.0, t2

        k = target_axis  # unit rotation axis
        sin_a = np.sin(angle_rad)
        cos_a = np.cos(angle_rad)

        # Skew-symmetric matrix K
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ])
        R2 = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

        return R2, angle_deg, t2

    @staticmethod
    def _computeAxisRotation(axis, angle_deg):
        """Compute 3x3 rotation matrix for rotation around a given axis.

        Args:
            axis: Unit axis vector (3D)
            angle_deg: Rotation angle in degrees

        Returns:
            3x3 rotation matrix (numpy)
        """
        k = axis / np.linalg.norm(axis)
        angle_rad = np.radians(angle_deg)
        sin_a = np.sin(angle_rad)
        cos_a = np.cos(angle_rad)

        # Skew-symmetric matrix K
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0],
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

        return R