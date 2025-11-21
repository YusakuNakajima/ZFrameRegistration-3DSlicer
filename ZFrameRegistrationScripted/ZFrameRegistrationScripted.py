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
            This module performs Z-frame registration.
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

        # Visualization Step (間引き表示用)
        self.visStepSpinBox = qt.QSpinBox()
        self.visStepSpinBox.setRange(1, 100)
        self.visStepSpinBox.setValue(1)
        self.visStepSpinBox.setToolTip("Reduce clutter by skipping slices during visualization.")
        parametersFormLayout.addRow("Visualization Step: ", self.visStepSpinBox)

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
        
        # 1. Apply Only Button
        self.applyButton = qt.QPushButton("Run Registration")
        self.applyButton.setToolTip("Run registration without visualizing points (Faster).")
        self.applyButton.setStyleSheet("font-weight: bold; padding: 6px;")
        self.applyButton.enabled = True
        actionsLayout.addWidget(self.applyButton)
        
        # 2. Apply & Visualize Button
        self.applyVisButton = qt.QPushButton("Run & Visualize Points")
        self.applyVisButton.setToolTip("Run registration and visualize detected points (Good for debugging).")
        self.applyVisButton.setStyleSheet("font-weight: bold; padding: 6px; color: #2e5e2e;")
        self.applyVisButton.enabled = True
        actionsLayout.addWidget(self.applyVisButton)
        
        # 3. Visualize Topology Button
        self.visualizeButton = qt.QPushButton("Visualize Topology")
        self.visualizeButton.toolTip = "Visualize the Z-frame topology definition."
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
        if node:
            dims = node.GetImageData().GetDimensions()
            num_slices = dims[2]
            self.sliceRangeWidget.maximum = num_slices
            self.sliceRangeWidget.minimum = 0
            self.sliceRangeWidget.minimumValue = 0
            self.sliceRangeWidget.maximumValue = num_slices

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
        self.runRegistration(visualize=False)

    def onApplyVisButton(self):
        self.runRegistration(visualize=True)

    def runRegistration(self, visualize):
        try:
            self.logic.run(self.inputSelector.currentNode(),
                     self.outputSelector.currentNode(),
                     self.zframeConfigSelector.currentText,
                     self.fiducialTypeSelector.currentText,
                     self.frameTopologyTextEdit.toPlainText(),
                     int(self.sliceRangeWidget.minimumValue),
                     int(self.sliceRangeWidget.maximumValue),
                     visualize=visualize,
                     visualizeStep=int(self.visStepSpinBox.value))
        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: "+str(e))
            import traceback
            traceback.print_exc()

class ZFrameRegistrationScriptedLogic(ScriptedLoadableModuleLogic):
    def run(self, inputVolume, outputTransform, zframeConfig, zframeType, frameTopology, startSlice, endSlice, visualize=False, visualizeStep=1):
        logging.info('Processing started')
        
        if not inputVolume or not outputTransform:
            raise ValueError("Input volume or output transform is missing")
            
        imageData = inputVolume.GetImageData()
        if not imageData:
            raise ValueError("Input image is invalid")
        
        dim = imageData.GetDimensions()
        imageDataArr = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
        imageDataArr = imageDataArr.reshape(dim[2], dim[1], dim[0]).transpose(2,1,0)

        origin = inputVolume.GetOrigin()
        spacing = inputVolume.GetSpacing()
        directions = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASDirectionMatrix(directions)

        imageTransform = np.eye(4)
        for i in range(3):
            for j in range(3):
                imageTransform[i,j] = spacing[j] * directions.GetElement(i,j)
            imageTransform[i,3] = origin[i]

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
            registration.SetInputImage(imageDataArr, imageTransform)
            registration.SetOrientationBase(ZquaternionBase)
            registration.SetFrameTopology(frameTopologyArr)
            
            result, Zposition, Zorientation, all_detected_points = registration.Register(sliceRange)
        
        # --- 検出点の可視化 ---
        if visualize and all_detected_points:
            points_to_visualize = all_detected_points[::visualizeStep]
            print(f"Visualizing points for {len(points_to_visualize)} slices (Step: {visualizeStep})")
            self.visualize_detected_points(inputVolume, points_to_visualize)
        elif not visualize:
            self.clear_detected_points_visualization()
        # -----------------------

        if result:
            matrix = zf.QuaternionToMatrix(Zorientation)
            zMatrix = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    zMatrix.SetElement(i,j, matrix[i][j])
                zMatrix.SetElement(i,3, Zposition[i])
            
            outputTransform.SetMatrixTransformToParent(zMatrix)
            logging.info('Processing completed')
            return True
        else:
            logging.error('Processing failed')
            return False

    def clear_detected_points_visualization(self):
        nodeName = "Detected Z-Frame Points"
        markupsNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if markupsNode:
            markupsNode.RemoveAllControlPoints()

    def visualize_detected_points(self, inputVolume, points_list):
        """
        points_list: [{"slice": int, "points": [[x,y],...]}, ...]
        """
        nodeName = "Detected Z-Frame Points"
        markupsNode = slicer.mrmlScene.GetFirstNodeByName(nodeName)
        if not markupsNode:
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", nodeName)
        
        markupsNode.RemoveAllControlPoints()
        markupsNode.GetDisplayNode().SetSelectedColor(0, 1, 0) # Green
        
        # ★サイズをハードコード (5.0)
        markupsNode.GetDisplayNode().SetTextScale(3.0) 
        markupsNode.GetDisplayNode().SetGlyphScale(3.0)
        
        # IJK to RAS Matrix
        ijkToRas = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRas)
        
        for item in points_list:
            slice_idx = item["slice"]
            points = item["points"]
            
            for i, pt in enumerate(points):
                ijk = [pt[0], pt[1], slice_idx, 1.0]
                ras = [0.0]*4
                ijkToRas.MultiplyPoint(ijk, ras)
                
                label = f"{slice_idx}-{i}"
                markupsNode.AddControlPoint(vtk.vtkVector3d(ras[0], ras[1], ras[2]), label)
            
        if points_list:
             last_pt = points_list[-1]["points"][0]
             last_slice = points_list[-1]["slice"]
             ijk = [last_pt[0], last_pt[1], last_slice, 1.0]
             ras = [0.0]*4
             ijkToRas.MultiplyPoint(ijk, ras)
             slicer.modules.markups.logic().JumpSlicesToLocation(ras[0], ras[1], ras[2], True)

    def visualize_topology(self, frameTopology):
        logging.info('Visualizing Z-frame topology with Lines and Labeled Points.')
        frameTopologyArr = []
        try:
            # トポロジー文字列のパース
            frameTopologyList = ''.join(frameTopology.split()).strip("[]").split("],[")
            for n in frameTopologyList:
                x, y, z = map(float, n.split(","))
                frameTopologyArr.append([x, y, z])
        except ValueError:
            slicer.util.errorDisplay("Topology format error.")
            return

        # 始点とベクトルを取得
        origins = frameTopologyArr[0:3]
        vectors = frameTopologyArr[3:6]
        
        # ---------------------------------------------------------
        # 1. 線を描画するためのモデルノード (Z-Frame Topology Lines)
        # ---------------------------------------------------------
        vtk_points_lines = vtk.vtkPoints()
        vtk_lines = vtk.vtkCellArray()
        
        pid = 0
        for o, v in zip(origins, vectors):
            start_pt = np.array(o)
            end_pt = np.array(o) + np.array(v)
            
            # 線の座標を追加
            vtk_points_lines.InsertNextPoint(start_pt)
            vtk_points_lines.InsertNextPoint(end_pt)
            
            # 線を作成
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
        
        # 線の色を「赤」に設定
        lineDisplayNode = lineModelNode.GetDisplayNode()
        if lineDisplayNode:
            lineDisplayNode.SetColor(1, 0, 0)  # 赤 (Red)
            lineDisplayNode.SetLineWidth(4.0)
            lineDisplayNode.SetOpacity(1.0)

        # ---------------------------------------------------------
        # 2. 名前付きの点を表示するためのマークアップノード (Z-Frame Topology Points)
        # ---------------------------------------------------------
        pointsNodeName = "Z-Frame Topology Points"
        markupsNode = slicer.mrmlScene.GetFirstNodeByName(pointsNodeName)
        if not markupsNode:
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", pointsNodeName)
        
        markupsNode.RemoveAllControlPoints()
        markupsNode.GetDisplayNode().SetSelectedColor(1, 0, 0) # 赤 (Selected)
        markupsNode.GetDisplayNode().SetColor(1, 0, 0)         # 赤 (Unselected)
        markupsNode.GetDisplayNode().SetTextScale(4.0)         # 文字サイズ
        markupsNode.GetDisplayNode().SetGlyphScale(3.0)        # 点のサイズ
        
        # 点を追加し、名前（ラベル）をつける
        # Zフレームのロッドは3本あるので、それぞれの始点(Start)と終点(End)に名前をつけます
        for i, (o, v) in enumerate(zip(origins, vectors)):
            start_pt = np.array(o)
            end_pt = np.array(o) + np.array(v)
            
            # 始点 (例: Rod1-Start)
            markupsNode.AddControlPoint(vtk.vtkVector3d(start_pt), f"Rod{i+1}-Start")
            
            # 終点 (例: Rod1-End)
            markupsNode.AddControlPoint(vtk.vtkVector3d(end_pt), f"Rod{i+1}-End")

        print(f"Created topology visualization: Lines and Labeled Points (Red)")