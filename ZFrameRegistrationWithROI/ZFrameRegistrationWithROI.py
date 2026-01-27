import os, sys
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import SimpleITK as sitk
import sitkUtils
from SlicerDevelopmentToolboxUtils.mixins import ModuleLogicMixin, ModuleWidgetMixin
from SlicerDevelopmentToolboxUtils.icons import Icons

# Add ZFrameRegistrationScripted path to import shared ZFrame/Registration module
moduleDir = os.path.dirname(os.path.realpath(__file__))
zframeScriptedDir = os.path.join(os.path.dirname(moduleDir), 'ZFrameRegistrationScripted')
if zframeScriptedDir not in sys.path:
    sys.path.append(zframeScriptedDir)

from ZFrame.Registration import zf, ZFrameRegistration


#
# ZFrameRegistrationWithROI
#

class OpenSourceZFrameRegistration(object):
  def __init__(self, mrmlScene, volume=None):
    self.inputVolume = volume
    self.mrmlScene = mrmlScene
    self.outputTransform = None
    self._setTransform()

  def setInputVolume(self, volume):
    self.inputVolume = volume
    self._setTransform()

  def _setTransform(self):
    if self.inputVolume:
      seriesNumber = self.inputVolume.GetName().split(":")[0]
      name = seriesNumber + "-ZFrameTransform"
      if self.outputTransform:
        self.mrmlScene.RemoveNode(self.outputTransform)
        self.outputTransform = None
      self.outputTransform = slicer.vtkMRMLLinearTransformNode()
      self.outputTransform.SetName(name)
      self.mrmlScene.AddNode(self.outputTransform)

  def runRegistration(self, start, end):
    if self.inputVolume:
      assert start != -1 and end != -1

      params = {'inputVolume': self.inputVolume, 'startSlice': start, 'endSlice': end,
                'outputTransform': self.outputTransform}
      slicer.cli.run(slicer.modules.zframeregistration, None, params, wait_for_completion=True)


class ZFrameRegistrationWithROI(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ZFrameRegistrationWithROI"  # TODO make this more human readable by adding spaces
    self.parent.categories = ["IGT"]
    self.parent.dependencies = []
    self.parent.contributors = ["Christian Herz (SPL), Longquan Chen (SPL), Junichi Tokuda (SPL), "
                                "Simon Di Maio (SPL), Andrey Fedorov (SPL). Updated by Mariana Bernardes (SPL)"]
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.

Update (Nov 22, 2023): Replaced AnnotationROINode (legacy) by MarkupsROINode.

"""  # replace with organization, grant and thanks.


#
# ZFrameRegistrationWithROIWidget
#

class ZFrameRegistrationWithROIWidget(ScriptedLoadableModuleWidget, ModuleWidgetMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)

  def onReload(self, moduleName="ZFrameRegistrationWithROI"):
    self.logic.cleanup()
    self.disconnectAll()
    slicer.mrmlScene.Clear(0)
    # Reload shared ZFrame.Registration module
    if 'ZFrame.Registration' in sys.modules:
      del sys.modules['ZFrame.Registration']
      print("ZFrame.Registration Deleted")
    ScriptedLoadableModuleWidget.onReload(self)

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = ZFrameRegistrationWithROILogic()
    self.setupSliceWidgets()
    # self.annotationLogic = slicer.modules.annotations.logic()
    self.markupsLogic = slicer.modules.markups.logic()
    self.zFrameRegistrationClass = OpenSourceZFrameRegistration
    self.roiObserverTag = None
    self.coverTemplateROI = None
    self.setupGUIAndConnections()

  def disconnectAll(self):
    try:
      self.zFrameTemplateVolumeSelector.disconnect('currentNodeChanged(bool)')
      self.retryZFrameRegistrationButton.clicked.disconnect()
      self.runZFrameRegistrationButton.clicked.disconnect()
      self.orientationSelector.disconnect('currentIndexChanged(int)')
      self.zframeConfigSelector.disconnect('currentTextChanged(QString)')
      self.runScriptedRegistrationButton.clicked.disconnect()
      self.visualizeDetectedPointsButton.clicked.disconnect()
      self.visualizeTopologyButton.clicked.disconnect()
    except Exception:
      pass

  def setupSliceWidgets(self):
    self.createSliceWidgetClassMembers("Red")
    self.createSliceWidgetClassMembers("Yellow")
    self.createSliceWidgetClassMembers("Green")

  def setupGUIAndConnections(self):
    self.zFrameTopologies = {}

    # Select zFrame model
    modelGroupBox = qt.QGroupBox()
    self.layout.addWidget(modelGroupBox)
    modelLayout = qt.QFormLayout(modelGroupBox)
    self.modelFileSelector = qt.QComboBox()
    modelPath= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Resources/zframe')
    self.modelList = [f for f in os.listdir(modelPath) if os.path.isfile(os.path.join(modelPath, f))]
    self.modelFileSelector.addItems(self.modelList)
    modelLayout.addRow('zFrame Model:',self.modelFileSelector)

    iconSize = qt.QSize(36, 36)
    self.inputVolumeGroupBox = qt.QGroupBox()
    self.inputVolumeGroupBoxLayout = qt.QFormLayout()
    self.inputVolumeGroupBox.setLayout(self.inputVolumeGroupBoxLayout)
    self.inputVolumeGroupBox.setFlat(True)
    self.zFrameTemplateVolumeSelector = self.createComboBox(nodeTypes=["vtkMRMLScalarVolumeNode", ""])
    self.inputVolumeGroupBoxLayout.addRow("ZFrame template volume: ", self.zFrameTemplateVolumeSelector)
    self.layout.addWidget(self.inputVolumeGroupBox)
    self.layout.addStretch()
    self.zFrameRegistrationManualIndexesGroupBox = qt.QGroupBox("Use manual start/end indexes")
    self.zFrameRegistrationManualIndexesGroupBox.setCheckable(True)
    self.zFrameRegistrationManualIndexesGroupBoxLayout = qt.QGridLayout()
    self.zFrameRegistrationManualIndexesGroupBox.setLayout(self.zFrameRegistrationManualIndexesGroupBoxLayout)
    self.zFrameRegistrationManualIndexesGroupBox.checked = False
    self.zFrameRegistrationStartIndex = qt.QSpinBox()
    self.zFrameRegistrationEndIndex = qt.QSpinBox()
    hBox = qt.QWidget()
    hBox.setLayout(qt.QHBoxLayout())
    hBox.layout().addWidget(qt.QLabel("start"))
    hBox.layout().addWidget(self.zFrameRegistrationStartIndex)
    hBox.layout().addWidget(qt.QLabel("end"))
    hBox.layout().addWidget(self.zFrameRegistrationEndIndex)
    self.zFrameRegistrationManualIndexesGroupBoxLayout.addWidget(hBox, 1, 1, qt.Qt.AlignRight)
    self.runZFrameRegistrationButton = self.createButton("", enabled=False, icon=Icons.apply, iconSize=iconSize,
                                                         toolTip="Run ZFrame Registration")
    self.retryZFrameRegistrationButton = self.createButton("", enabled=False, icon=Icons.retry, iconSize=iconSize,
                                                           toolTip="Reset")
    self.layout.addWidget(self.zFrameRegistrationManualIndexesGroupBox)
    widget = qt.QWidget()
    widget.setLayout(qt.QHBoxLayout())
    widget.layout().addWidget(self.runZFrameRegistrationButton)
    widget.layout().addWidget(self.retryZFrameRegistrationButton)
    self.layout.addWidget(widget)

    # ========== Scripted Registration Parameters (from ZFrameRegistrationScripted) ==========
    scriptedParamsCollapsibleButton = ctk.ctkCollapsibleButton()
    scriptedParamsCollapsibleButton.text = "Scripted Registration Parameters"
    scriptedParamsCollapsibleButton.collapsed = True
    self.layout.addWidget(scriptedParamsCollapsibleButton)
    scriptedParamsLayout = qt.QFormLayout(scriptedParamsCollapsibleButton)

    # Orientation Selector
    self.orientationSelector = qt.QComboBox()
    self.orientationSelector.addItems(["Axial (Red)", "Coronal (Green)", "Sagittal (Yellow)"])
    self.orientationSelector.setToolTip("Select the slice orientation where Z-frame dots are visible.")
    scriptedParamsLayout.addRow("Detection Orientation: ", self.orientationSelector)
    self.orientationSelector.connect("currentIndexChanged(int)", self.onOrientationChanged)

    # Fiducial type selector
    self.fiducialTypeSelector = qt.QComboBox()
    self.fiducialTypeSelector.addItems(["7-fiducial", "9-fiducial"])
    scriptedParamsLayout.addRow("Fiducial Frame Type: ", self.fiducialTypeSelector)
    self.fiducialTypeSelector.setCurrentIndex(0)

    # Z-frame configuration selector
    self.zframeConfigSelector = qt.QComboBox()
    self.loadZFrameConfigs()
    scriptedParamsLayout.addRow("Z-Frame Configuration: ", self.zframeConfigSelector)
    self.zframeConfigSelector.setCurrentIndex(3) if self.zframeConfigSelector.count > 3 else None
    self.zframeConfigSelector.connect("currentTextChanged(QString)", self.onZFrameConfigChanged)

    # Frame Topology Text Edit
    self.frameTopologyTextEdit = qt.QTextEdit()
    self.frameTopologyTextEdit.setReadOnly(False)
    self.frameTopologyTextEdit.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    self.frameTopologyTextEdit.setMaximumHeight(40)
    scriptedParamsLayout.addRow("Frame Topology: ", self.frameTopologyTextEdit)
    self.onZFrameConfigChanged(self.zframeConfigSelector.currentText)

    # Slice range
    self.sliceRangeWidget = slicer.qMRMLRangeWidget()
    self.sliceRangeWidget.decimals = 0
    self.sliceRangeWidget.minimum = 0
    self.sliceRangeWidget.maximum = 100
    self.sliceRangeWidget.singleStep = 1
    scriptedParamsLayout.addRow("Slice Range: ", self.sliceRangeWidget)

    # Visualization Step
    self.visStepSpinBox = qt.QSpinBox()
    self.visStepSpinBox.setRange(1, 100)
    self.visStepSpinBox.setValue(1)
    self.visStepSpinBox.setToolTip("Reduce clutter by skipping slices during visualization.")
    scriptedParamsLayout.addRow("Visualization Step: ", self.visStepSpinBox)

    # Marker Diameter Setting
    self.markerDiameterSpinBox = qt.QSpinBox()
    self.markerDiameterSpinBox.setRange(3, 100)
    self.markerDiameterSpinBox.setValue(11)
    self.markerDiameterSpinBox.setToolTip("Diameter of the fiducial marker in pixels.")
    scriptedParamsLayout.addRow("Marker Diameter (px): ", self.markerDiameterSpinBox)

    # Output transform selector
    self.outputTransformSelector = slicer.qMRMLNodeComboBox()
    self.outputTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.outputTransformSelector.selectNodeUponCreation = True
    self.outputTransformSelector.addEnabled = True
    self.outputTransformSelector.removeEnabled = True
    self.outputTransformSelector.noneEnabled = True
    self.outputTransformSelector.showHidden = False
    self.outputTransformSelector.showChildNodeTypes = False
    self.outputTransformSelector.setMRMLScene(slicer.mrmlScene)
    self.outputTransformSelector.setToolTip("Pick the output transform")
    scriptedParamsLayout.addRow("Output Transform: ", self.outputTransformSelector)

    # --- Scripted Actions Area ---
    baseButtonStyle = "font-weight: bold; font-size: 13px; padding: 6px;"

    self.runScriptedRegistrationButton = qt.QPushButton("Run Scripted Registration")
    self.runScriptedRegistrationButton.setToolTip("Run registration using Python script (Compute transform).")
    self.runScriptedRegistrationButton.setStyleSheet(baseButtonStyle)
    self.runScriptedRegistrationButton.enabled = True
    scriptedParamsLayout.addRow(self.runScriptedRegistrationButton)

    self.visualizeDetectedPointsButton = qt.QPushButton("Visualize Detected Points")
    self.visualizeDetectedPointsButton.setToolTip("Detect and visualize points ONLY (Does not update transform).")
    self.visualizeDetectedPointsButton.setStyleSheet(baseButtonStyle + " color: #00aa00;")
    self.visualizeDetectedPointsButton.enabled = True
    scriptedParamsLayout.addRow(self.visualizeDetectedPointsButton)

    self.visualizeTopologyButton = qt.QPushButton("Visualize Frame Topology")
    self.visualizeTopologyButton.setToolTip("Visualize the Z-frame topology definition.")
    self.visualizeTopologyButton.setStyleSheet(baseButtonStyle + " color: #d60000;")
    self.visualizeTopologyButton.enabled = True
    scriptedParamsLayout.addRow(self.visualizeTopologyButton)

    # Connect Scripted Buttons
    self.runScriptedRegistrationButton.connect('clicked(bool)', self.onRunScriptedRegistrationButton)
    self.visualizeDetectedPointsButton.connect('clicked(bool)', self.onVisualizeDetectedPointsButton)
    self.visualizeTopologyButton.connect('clicked(bool)', self.onVisualizeTopologyButton)

    self.layout.addStretch(1)
    self.zFrameTemplateVolumeSelector.connect('currentNodeChanged(bool)', self.loadVolumeAndEnableEditor)
    self.retryZFrameRegistrationButton.clicked.connect(self.onRetryZFrameRegistrationButtonClicked)
    self.runZFrameRegistrationButton.clicked.connect(self.onApplyZFrameRegistrationButtonClicked)
    
  def loadVolumeAndEnableEditor(self):
    zFrameTemplateVolume = self.zFrameTemplateVolumeSelector.currentNode()
    if zFrameTemplateVolume:
      self.logic.templateVolume = zFrameTemplateVolume
      self.activateZFrameRegistration()
      # Update slice range for scripted registration
      self.onOrientationChanged(self.orientationSelector.currentIndex)
    else:
      self.logic.templateVolume = None
      self.resetZFrameRegistration()
      self.setROIMode(False)
      self.setBackgroundAndForegroundIDs(foregroundVolumeID=None, backgroundVolumeID=None)

  def activateZFrameRegistration(self):
    self.zFrameRegistrationManualIndexesGroupBox.checked = False
    if self.logic.templateVolume:
      self.resetZFrameRegistration()
      self.setBackgroundAndForegroundIDs(foregroundVolumeID=None, backgroundVolumeID=self.logic.templateVolume.GetID())
      self.redSliceNode.SetSliceVisible(True)
      if self.zFrameRegistrationClass is OpenSourceZFrameRegistration:
        self.addROIObserver()
        self.setROIMode(True)

  def resetZFrameRegistration(self):
    self.logic.clearVolumeNodes()
    if self.coverTemplateROI:
      slicer.mrmlScene.RemoveNode(self.coverTemplateROI)
      self.coverTemplateROI = None
    self.runZFrameRegistrationButton.enabled = False
    self.retryZFrameRegistrationButton.enabled = False
    if self.logic.zFrameModelNode:
      self.logic.zFrameModelNode.GetDisplayNode().SetSliceIntersectionVisibility(False)
      self.logic.zFrameModelNode.SetDisplayVisibility(False)
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

  def setBackgroundAndForegroundIDs(self, foregroundVolumeID, backgroundVolumeID):
    self.redCompositeNode.SetForegroundVolumeID(foregroundVolumeID)
    self.redCompositeNode.SetBackgroundVolumeID(backgroundVolumeID)
    self.redSliceNode.SetOrientationToAxial()
    self.yellowCompositeNode.SetForegroundVolumeID(foregroundVolumeID)
    self.yellowCompositeNode.SetBackgroundVolumeID(backgroundVolumeID)
    self.yellowSliceNode.SetOrientationToSagittal()
    self.greenCompositeNode.SetForegroundVolumeID(foregroundVolumeID)
    self.greenCompositeNode.SetBackgroundVolumeID(backgroundVolumeID)
    self.greenSliceNode.SetOrientationToCoronal()

  def addROIObserver(self):
    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAdded(caller, event, calldata):
      node = calldata
      # if isinstance(node, slicer.vtkMRMLAnnotationROINode) :
      if isinstance(node, slicer.vtkMRMLMarkupsROINode) : #Mariana
        self.removeROIObserver()
        self.coverTemplateROI = node
        self.runZFrameRegistrationButton.enabled = self.isRegistrationPossible()

    if self.roiObserverTag:
      self.removeROIObserver()
    self.roiObserverTag = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, onNodeAdded)

  def isRegistrationPossible(self):
    return self.coverTemplateROI is not None and self.logic.templateVolume

  def removeROIObserver(self):
    if self.roiObserverTag:
      self.roiObserverTag = slicer.mrmlScene.RemoveObserver(self.roiObserverTag)

  def setROIMode(self, mode):
    if mode == False:
      self.markupsLogic.StopPlaceMode(False)
    else:
      self.markupsLogic.StartPlaceMode(False)
    mrmlScene = self.markupsLogic.GetMRMLScene()
    selectionNode = mrmlScene.GetNthNodeByClass(0, "vtkMRMLSelectionNode")
    # selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLAnnotationROINode")
    selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsROINode") # Mariana    

  def onApplyZFrameRegistrationButtonClicked(self):
    self.retryZFrameRegistrationButton.enabled = True
    zFrameTemplateVolume = self.logic.templateVolume
    zFrameModelName = self.modelFileSelector.currentText
    try:
      # self.annotationLogic.SetAnnotationLockedUnlocked(self.coverTemplateROI.GetID())
      self.markupsLogic.ToggleAllControlPointsLocked(self.coverTemplateROI) # Mariana
      
      if not self.zFrameRegistrationManualIndexesGroupBox.checked:
        self.logic.runZFrameOpenSourceRegistration(zFrameModelName, zFrameTemplateVolume, self.coverTemplateROI)
        self.zFrameRegistrationStartIndex.value = self.logic.startIndex
        self.zFrameRegistrationEndIndex.value = self.logic.endIndex
      else:
        startIndex = self.zFrameRegistrationStartIndex.value
        endIndex = self.zFrameRegistrationEndIndex.value
        self.logic.runZFrameOpenSourceRegistration(zFrameModelName, zFrameTemplateVolume, self.coverTemplateROI, start=startIndex,
                                                   end=endIndex)
      self.setBackgroundAndForegroundIDs(foregroundVolumeID=None, backgroundVolumeID=self.logic.templateVolume.GetID())
      self.logic.zFrameModelNode.SetAndObserveTransformNodeID(self.logic.openSourceRegistration.outputTransform.GetID())
      self.logic.zFrameModelNode.GetDisplayNode().SetSliceIntersectionVisibility(True)
      self.logic.zFrameModelNode.SetDisplayVisibility(True)
    except AttributeError as exc:
      slicer.util.errorDisplay("An error occurred. For further information click 'Show Details...'",
                               windowTitle=self.__class__.__name__, detailedText=str(exc.message))

  def onRetryZFrameRegistrationButtonClicked(self):
    self.activateZFrameRegistration()

  # ========== Scripted Registration Methods (from ZFrameRegistrationScripted) ==========
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

  def onOrientationChanged(self, index):
    node = self.zFrameTemplateVolumeSelector.currentNode()
    if node:
      dims = node.GetImageData().GetDimensions()
      orientation = self.orientationSelector.currentText
      max_slice = 0
      if "Axial" in orientation:
        max_slice = dims[2]
      elif "Coronal" in orientation:
        max_slice = dims[1]
      elif "Sagittal" in orientation:
        max_slice = dims[0]
      self.sliceRangeWidget.maximum = max_slice
      self.sliceRangeWidget.minimum = 0
      self.sliceRangeWidget.minimumValue = 0
      self.sliceRangeWidget.maximumValue = max_slice

  def onRunScriptedRegistrationButton(self):
    self.runScriptedRegistration(visualize=False, updateTransform=True)

  def onVisualizeDetectedPointsButton(self):
    self.runScriptedRegistration(visualize=True, updateTransform=False)

  def onVisualizeTopologyButton(self):
    try:
      self.logic.visualize_topology(self.frameTopologyTextEdit.toPlainText())
    except Exception as e:
      slicer.util.errorDisplay("Failed to visualize topology: "+str(e))
      import traceback
      traceback.print_exc()

  def runScriptedRegistration(self, visualize, updateTransform=True):
    try:
      inputVolume = self.zFrameTemplateVolumeSelector.currentNode()
      outputTransform = self.outputTransformSelector.currentNode()
      if not inputVolume:
        slicer.util.errorDisplay("Please select an input volume.")
        return
      if not outputTransform and updateTransform:
        slicer.util.errorDisplay("Please select an output transform.")
        return
      self.logic.runScriptedRegistration(
        inputVolume,
        outputTransform,
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
      # Apply transform to Z-frame model if available
      if updateTransform and outputTransform and self.logic.zFrameModelNode:
        self.logic.zFrameModelNode.SetAndObserveTransformNodeID(outputTransform.GetID())
        self.logic.zFrameModelNode.GetDisplayNode().SetSliceIntersectionVisibility(True)
        self.logic.zFrameModelNode.SetDisplayVisibility(True)
    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# ZFrameRegistrationWithROILogic
#

class ZFrameRegistrationWithROILogic(ScriptedLoadableModuleLogic, ModuleLogicMixin):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  # ZFRAME_MODEL_PATH = 'zframe-model.vtk'
  # ZFRAME_MODEL_NAME = 'ZFrameModel'

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.redSliceWidget = slicer.app.layoutManager().sliceWidget("Red")
    self.redSliceView = self.redSliceWidget.sliceView()
    self.redSliceLogic = self.redSliceWidget.sliceLogic()
    self.otsuFilter = sitk.OtsuThresholdImageFilter()
    self.openSourceRegistration = OpenSourceZFrameRegistration(slicer.mrmlScene)
    self.templateVolume = None
    self.zFrameCroppedVolume = None
    self.zFrameLabelVolume = None
    self.zFrameMaskedVolume = None
    self.otsuOutputVolume = None
    self.startIndex = None
    self.endIndex = None
    self.zFrameModelNode = None
    self.resetAndInitializeData()

  def resetAndInitializeData(self):
    self.cleanup()
    self.startIndex = None
    self.endIndex = None

  def clearVolumeNodes(self):
    if self.zFrameCroppedVolume:
      slicer.mrmlScene.RemoveNode(self.zFrameCroppedVolume)
      self.zFrameCroppedVolume = None
    if self.zFrameLabelVolume:
      slicer.mrmlScene.RemoveNode(self.zFrameLabelVolume)
      self.zFrameLabelVolume = None
    if self.zFrameMaskedVolume:
      slicer.mrmlScene.RemoveNode(self.zFrameMaskedVolume)
      self.zFrameMaskedVolume = None
    if self.otsuOutputVolume:
      slicer.mrmlScene.RemoveNode(self.otsuOutputVolume)
      self.otsuOutputVolume = None

  def cleanup(self):
    self.clearVolumeNodes()
    self.clearOldCalculationNodes()

  def clearOldCalculationNodes(self):
    if self.openSourceRegistration.inputVolume:
      slicer.mrmlScene.RemoveNode(self.openSourceRegistration.inputVolume)
      self.openSourceRegistration.inputVolume = None
    if self.zFrameModelNode:
      slicer.mrmlScene.RemoveNode(self.zFrameModelNode)
      self.zFrameModelNode = None
    if self.openSourceRegistration.outputTransform:
      slicer.mrmlScene.RemoveNode(self.openSourceRegistration.outputTransform)
      self.openSourceRegistration.outputTransform = None

  def loadZFrameModel(self, zFrameModelName):
    if self.zFrameModelNode:
      slicer.mrmlScene.RemoveNode(self.zFrameModelNode)
      self.zFrameModelNode = None
    currentFilePath = os.path.dirname(os.path.realpath(__file__))
    zFrameModelPath = os.path.join(currentFilePath, "Resources", "zframe", zFrameModelName)
    _, self.zFrameModelNode = slicer.util.loadModel(zFrameModelPath, returnNode=True)
    self.zFrameModelNode.SetName('ZFrameModel')
    modelDisplayNode = self.zFrameModelNode.GetDisplayNode()
    modelDisplayNode.SetColor(1, 1, 0)
    self.zFrameModelNode.SetDisplayVisibility(False)

  def runZFrameOpenSourceRegistration(self, zFrameModelName, zFrameTemplateVolume, coverTemplateROI, start=None, end=None):
    self.startIndex = start
    self.endIndex = end
    self.loadZFrameModel(zFrameModelName) # Load selected zFrame Model
    self.zFrameCroppedVolume = self.createCroppedVolume(zFrameTemplateVolume, coverTemplateROI)
    self.zFrameLabelVolume = self.createLabelMapFromCroppedVolume(self.zFrameCroppedVolume, "labelmap")
    self.zFrameMaskedVolume = self.createMaskedVolume(zFrameTemplateVolume, self.zFrameLabelVolume,
                                                      outputVolumeName="maskedTemplateVolume")
    self.zFrameMaskedVolume.SetName(zFrameTemplateVolume.GetName() + "-label")
    if self.startIndex is None or self.endIndex is None:
      self.startIndex, center, self.endIndex = self.getROIMinCenterMaxSliceNumbers(coverTemplateROI)
      self.otsuOutputVolume = self.applyITKOtsuFilter(self.zFrameMaskedVolume)
      self.dilateMask(self.otsuOutputVolume)
      self.startIndex, self.endIndex = self.getStartEndWithConnectedComponents(self.otsuOutputVolume, center)
    self.openSourceRegistration.setInputVolume(self.zFrameMaskedVolume)
    self.openSourceRegistration.runRegistration(self.startIndex, self.endIndex)
    self.clearVolumeNodes()
    return True

  def getROIMinCenterMaxSliceNumbers(self, coverTemplateROI):
    center = [0.0, 0.0, 0.0]
    coverTemplateROI.GetXYZ(center)
    bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    coverTemplateROI.GetRASBounds(bounds)
    pMin = [bounds[0], bounds[2], bounds[4]]
    pMax = [bounds[1], bounds[3], bounds[5]]
    return [self.getIJKForXYZ(self.redSliceWidget, pMin)[2], self.getIJKForXYZ(self.redSliceWidget, center)[2],
            self.getIJKForXYZ(self.redSliceWidget, pMax)[2]]

  def getStartEndWithConnectedComponents(self, volume, center):
    address = sitkUtils.GetSlicerITKReadWriteAddress(volume.GetName())
    image = sitk.ReadImage(address)
    start = self.getStartSliceUsingConnectedComponents(center, image)
    end = self.getEndSliceUsingConnectedComponents(center, image)
    return start, end

  def getStartSliceUsingConnectedComponents(self, center, image):
    sliceIndex = start = center
    while sliceIndex > 0:
      if self.getIslandCount(image, sliceIndex) > 6:
        start = sliceIndex
        sliceIndex -= 1
        continue
      break
    return start

  def getEndSliceUsingConnectedComponents(self, center, image):
    imageSize = image.GetSize()
    sliceIndex = end = center
    while sliceIndex < imageSize[2]:
      if self.getIslandCount(image, sliceIndex) > 6:
        end = sliceIndex
        sliceIndex += 1
        continue
      break
    return end

  def applyITKOtsuFilter(self, volume):
    inputVolume = sitk.Cast(sitkUtils.PullVolumeFromSlicer(volume.GetID()), sitk.sitkInt16)
    self.otsuFilter.SetInsideValue(0)
    self.otsuFilter.SetOutsideValue(1)
    otsuITKVolume = self.otsuFilter.Execute(inputVolume)
    # return sitkUtils.PushToSlicer(otsuITKVolume, "otsuITKVolume", 0, True)
    return sitkUtils.PushVolumeToSlicer(otsuITKVolume, name="otsuITKVolume") # Mariana fix

  # ========== Scripted Registration Methods (from ZFrameRegistrationScripted) ==========
  def runScriptedRegistration(self, inputVolume, outputTransform, zframeConfig, zframeType, frameTopology, startSlice, endSlice, visualize=False, visualizeStep=1, markerDiameter=11, updateTransform=True, orientation="Axial"):
    logging.info('Scripted Registration Processing started')

    if not inputVolume:
      raise ValueError("Input volume is missing")
    if not outputTransform and updateTransform:
      raise ValueError("Output transform is missing")

    imageData = inputVolume.GetImageData()
    if not imageData:
      raise ValueError("Input image is invalid")

    dim = imageData.GetDimensions()
    imageDataArr = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
    imageDataArr = imageDataArr.reshape(dim[2], dim[1], dim[0])

    origin = inputVolume.GetOrigin()
    spacing = inputVolume.GetSpacing()
    directions = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASDirectionMatrix(directions)

    imageTransform = np.eye(4)
    for i in range(3):
      for j in range(3):
        imageTransform[i,j] = spacing[j] * directions.GetElement(i,j)
      imageTransform[i,3] = origin[i]

    # Handle Orientation Swapping
    if "Axial" in orientation:
      imageDataArr = imageDataArr.transpose(2,1,0)
    elif "Coronal" in orientation:
      imageDataArr = imageDataArr.transpose(2, 0, 1)
      original_matrix = imageTransform.copy()
      imageTransform[:, 0] = original_matrix[:, 0]
      imageTransform[:, 1] = original_matrix[:, 2]
      imageTransform[:, 2] = original_matrix[:, 1]
    elif "Sagittal" in orientation:
      imageDataArr = imageDataArr.transpose(1, 0, 2)
      original_matrix = imageTransform.copy()
      imageTransform[:, 0] = original_matrix[:, 1]
      imageTransform[:, 1] = original_matrix[:, 2]
      imageTransform[:, 2] = original_matrix[:, 0]

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

      result, Zposition, Zorientation, all_detected_points, rms_error = registration.Register(sliceRange)

    # Visualization of detected points
    if visualize and all_detected_points:
      points_to_visualize = all_detected_points[::visualizeStep]
      print(f"Visualizing points for {len(points_to_visualize)} slices (Step: {visualizeStep})")
      self.visualize_detected_points(inputVolume, points_to_visualize, orientation)

    if result and updateTransform:
      matrix = zf.QuaternionToMatrix(Zorientation)
      zMatrix = vtk.vtkMatrix4x4()
      for i in range(3):
        for j in range(3):
          zMatrix.SetElement(i,j, matrix[i][j])
        zMatrix.SetElement(i,3, Zposition[i])

      outputTransform.SetMatrixTransformToParent(zMatrix)
      logging.info('Scripted Registration Processing completed')
      if rms_error is not None:
        rms_msg = f"RMS Error: {rms_error:.4f} mm"
        logging.info(rms_msg)
        slicer.util.infoDisplay(rms_msg, windowTitle="Registration Result")
      return True
    elif result and not updateTransform:
      logging.info('Detection completed (Transform update skipped)')
      if rms_error is not None:
        rms_msg = f"RMS Error: {rms_error:.4f} mm"
        logging.info(rms_msg)
        slicer.util.infoDisplay(rms_msg, windowTitle="Detection Result")
      return True
    else:
      logging.error('Scripted Registration Processing failed')
      slicer.util.errorDisplay("Z-Frame registration failed. No valid slices found.\nPlease check:\n1. 'Scripted Registration Parameters' are used.\n2. Marker Diameter is correct.\n3. Slice Range covers the frame.\n4. View 'Python Interactor' for debug logs.")
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
    markupsNode.GetDisplayNode().SetSelectedColor(0, 1, 0)

    markupsNode.GetDisplayNode().SetTextScale(3.0)
    markupsNode.GetDisplayNode().SetGlyphScale(3.0)

    ijkToRas = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(ijkToRas)

    for item in points_list:
      slice_idx = item["slice"]
      points = item["points"]

      for i, pt in enumerate(points):
        if "Axial" in orientation:
          ijk = [pt[0], pt[1], slice_idx, 1.0]
        elif "Coronal" in orientation:
          ijk = [pt[0], slice_idx, pt[1], 1.0]
        elif "Sagittal" in orientation:
          ijk = [slice_idx, pt[0], pt[1], 1.0]

        ras = [0.0]*4
        ijkToRas.MultiplyPoint(ijk, ras)

        label = f"{slice_idx}-{i}"
        markupsNode.AddControlPoint(vtk.vtkVector3d(ras[0], ras[1], ras[2]), label)

    if points_list:
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
      frameTopologyList = ''.join(frameTopology.split()).strip("[]").split("],[")
      for n in frameTopologyList:
        x, y, z = map(float, n.split(","))
        frameTopologyArr.append([x, y, z])
    except ValueError:
      slicer.util.errorDisplay("Topology format error.")
      return

    origins = frameTopologyArr[0:3]
    vectors = frameTopologyArr[3:6]

    vtk_points_lines = vtk.vtkPoints()
    vtk_lines = vtk.vtkCellArray()

    pid = 0
    for o, v in zip(origins, vectors):
      start_pt = np.array(o)
      end_pt = np.array(o) + np.array(v)

      vtk_points_lines.InsertNextPoint(start_pt)
      vtk_points_lines.InsertNextPoint(end_pt)

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

    lineDisplayNode = lineModelNode.GetDisplayNode()
    if lineDisplayNode:
      lineDisplayNode.SetColor(1, 0, 0)
      lineDisplayNode.SetLineWidth(4.0)
      lineDisplayNode.SetOpacity(1.0)

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
    

class ZFrameRegistrationWithROITest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  groundTruthMatrix = [0.9999315859310454, 0.009689047677719153, -0.006549676681617225, 5.971096704891779,
                       -0.009774406649458021, 0.9998660159742193, -0.013128544923338871, -18.918600331582244,
                       0.006421595844729844, 0.013191666276940213, 0.999892377445857, 102.1792443094631,
                       0.0, 0.0, 0.0, 1.0]
  
  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_ZFrameRegistrationWithROI1()

  def isclose(self, a, b, rel_tol=1e-05, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

  def test_ZFrameRegistrationWithROI1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #

    currentFilePath = os.path.dirname(os.path.realpath(__file__))
    imageDataPath = os.path.join(os.path.abspath(os.path.join(currentFilePath, os.pardir)), "ZFrameRegistration",
                                 "Data", "Input", "CoverTemplateMasked.nrrd")
    print(imageDataPath)
    _, imageDataNode = slicer.util.loadVolume(imageDataPath, returnNode=True)
    slicer.app.processEvents()
    self.delayDisplay('Finished with loading')

    zFrameRegistrationLogic = ZFrameRegistrationWithROILogic()
    # ROINode = slicer.vtkMRMLAnnotationROINode()
    ROINode = slicer.vtkMRMLMarkupsROINode() #  Mariana  
    ROINode.SetName("ROINodeForCropping")
    ROICenterPoint = [-6.91920280456543, 15.245062828063965, -101.13504791259766]
    ROINode.SetXYZ(ROICenterPoint)
    ROIRadiusXYZ = [36.46055603027344, 38.763328552246094, 36.076759338378906]
    ROINode.SetRadiusXYZ(ROIRadiusXYZ)
    slicer.mrmlScene.AddNode(ROINode)
    slicer.app.processEvents()

    zFrameRegistrationLogic.runZFrameOpenSourceRegistration('zframe_original_vertical.vtk', imageDataNode, coverTemplateROI=ROINode)
    slicer.app.processEvents()
    transformNode = zFrameRegistrationLogic.openSourceRegistration.outputTransform
    transformMatrix = transformNode.GetTransformFromParent().GetMatrix()
    testResultMatrix = [0.0] * 16
    transformMatrix.DeepCopy(testResultMatrix, transformMatrix)
    for index in range(len(self.groundTruthMatrix)):
      self.assertEqual(self.isclose(float(testResultMatrix[index]), float(self.groundTruthMatrix[index])), True)
    zFrameRegistrationLogic.clearVolumeNodes()

    self.delayDisplay('Test passed!')


class ZFrameRegistrationWithROISlicelet(qt.QWidget):
  def __init__(self):
    qt.QWidget.__init__(self)
    self.setLayout(qt.QVBoxLayout())
    self.mainWidget = qt.QWidget()
    self.mainWidget.objectName = "qSlicerAppMainWindow"
    self.mainWidget.setLayout(qt.QVBoxLayout())

    self.setupLayoutWidget()

    self.moduleFrame = qt.QWidget()
    self.moduleFrameLayout = qt.QVBoxLayout()
    self.moduleFrame.setLayout(self.moduleFrameLayout)

    self.buttons = qt.QFrame()
    self.buttons.setLayout(qt.QHBoxLayout())
    self.moduleFrameLayout.addWidget(self.buttons)
    self.addDataButton = qt.QPushButton("Add Data")
    self.buttons.layout().addWidget(self.addDataButton)
    self.addDataButton.connect("clicked()", slicer.app.ioManager().openAddDataDialog)
    self.loadSceneButton = qt.QPushButton("Load Scene")
    self.buttons.layout().addWidget(self.loadSceneButton)
    self.loadSceneButton.connect("clicked()", slicer.app.ioManager().openLoadSceneDialog)

    self.zFrameRegistrationWidget = ZFrameRegistrationWithROIWidget(self.moduleFrame)
    self.zFrameRegistrationWidget.setup()
    self.zFrameRegistrationWidget.reloadCollapsibleButton.visible = False

    # TODO: resize self.widget.parent to minimum possible width

    self.scrollArea = qt.QScrollArea()
    self.scrollArea.setWidget(self.zFrameRegistrationWidget.parent)
    self.scrollArea.setWidgetResizable(True)
    self.scrollArea.setMinimumWidth(self.zFrameRegistrationWidget.parent.minimumSizeHint.width())

    self.splitter = qt.QSplitter()
    self.splitter.setOrientation(qt.Qt.Horizontal)
    self.splitter.addWidget(self.scrollArea)
    self.splitter.addWidget(self.layoutWidget)
    self.splitter.splitterMoved.connect(self.onSplitterMoved)

    self.splitter.setStretchFactor(0, 0)
    self.splitter.setStretchFactor(1, 1)
    self.splitter.handle(1).installEventFilter(self)

    self.mainWidget.layout().addWidget(self.splitter)
    self.mainWidget.show()

  def setupLayoutWidget(self):
    self.layoutWidget = qt.QWidget()
    self.layoutWidget.setLayout(qt.QHBoxLayout())
    layoutWidget = slicer.qMRMLLayoutWidget()
    layoutManager = slicer.qSlicerLayoutManager()
    layoutManager.setMRMLScene(slicer.mrmlScene)
    layoutManager.setScriptedDisplayableManagerDirectory(slicer.app.slicerHome + "/bin/Python/mrmlDisplayableManager")
    layoutWidget.setLayoutManager(layoutManager)
    slicer.app.setLayoutManager(layoutManager)
    layoutWidget.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    self.layoutWidget.layout().addWidget(layoutWidget)

  def eventFilter(self, obj, event):
    if event.type() == qt.QEvent.MouseButtonDblClick:
      self.onSplitterClick()

  def onSplitterMoved(self, pos, index):
    vScroll = self.scrollArea.verticalScrollBar()
    vScrollbarWidth = 4 if not vScroll.isVisible() else vScroll.width + 4
    if self.scrollArea.minimumWidth != self.zFrameRegistrationWidget.parent.minimumSizeHint.width() + vScrollbarWidth:
      self.scrollArea.setMinimumWidth(self.zFrameRegistrationWidget.parent.minimumSizeHint.width() + vScrollbarWidth)

  def onSplitterClick(self):
    if self.splitter.sizes()[0] > 0:
      self.splitter.setSizes([0, self.splitter.sizes()[1]])
    else:
      minimumWidth = self.zFrameRegistrationWidget.parent.minimumSizeHint.width()
      self.splitter.setSizes([minimumWidth, self.splitter.sizes()[1] - minimumWidth])


if __name__ == "__main__":
  import sys

  print(sys.argv)

  slicelet = ZFrameRegistrationWithROISlicelet()        
