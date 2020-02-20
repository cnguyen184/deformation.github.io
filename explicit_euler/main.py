import wx

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from wx import glcanvas

import ClothObject


class MyFrame(wx.Frame):
    def __init__(self):
        self.size = (1280, 720)
        wx.Frame.__init__(self, None, title='wx frame', size=self.size,
            style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        self.panel = MyPanel(self)


class MyPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        # OpenGL canvas within panel
        self.canvas = OpenGLCanvas(self)
        
        self.bAnimation = False
        self.ResetButton = wx.Button(self, wx.ID_ANY, 'Mass-Spring Reset', pos=(1030, 20),
                                     size=(200, 40), style=0)
        self.AnimationButton = wx.Button(self, wx.ID_ANY, 'Animate/Stop', pos=(1030, 60),
                                         size=(200, 40), style=0)

        # Slider for stiffness
        self.stiffnessLabel = wx.StaticText(self, -1, pos=(1030, 150), style=wx.ALIGN_CENTER)
        self.stiffnessLabel.SetLabel('Stiffness: ' + str(self.canvas.clothObject.stiffness))
        self.stiffnessSlider = wx.Slider(self, -1, pos=(1030, 180), size=(200, 50),
                                         style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                         value=1, minValue=1, maxValue=30)
        # Slider for step
        self.stepLabel = wx.StaticText(self, -1, pos=(1030, 250), style=wx.ALIGN_CENTER)
        self.stepLabel.SetLabel('Time Interval: ' + str(self.canvas.stepSize))
        self.stepSlider = wx.Slider(self, -1, pos=(1030, 280), size=(200, 50),
                                    style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                    value=10, minValue=1, maxValue=300)
        # Slider for damping
        self.dampLabel = wx.StaticText(self, -1, pos=(1030, 350), style=wx.ALIGN_CENTER)
        self.dampLabel.SetLabel('Damping Coeff.: ' + str(self.canvas.clothObject.damp))
        self.dampSlider = wx.Slider(self, -1, pos=(1030, 380), size=(200, 50),
                                    style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS,
                                    value=0, minValue=0, maxValue=30)

        # Callback functions
        self.Bind(wx.EVT_BUTTON, self.OnAnimationButton, self.AnimationButton)
        self.Bind(wx.EVT_BUTTON, self.OnResetButton, self.ResetButton)
        self.Bind(wx.EVT_SLIDER, self.OnStiffnessSlider, self.stiffnessSlider)
        self.Bind(wx.EVT_SLIDER, self.OnStepSlider, self.stepSlider)
        self.Bind(wx.EVT_SLIDER, self.OnDampSlider, self.dampSlider)

    def OnAnimationButton(self, event):
        """Toggles animation flag"""
        if self.bAnimation is False:
            self.bAnimation = True
        else:
            self.bAnimation = False
        self.canvas.bAnimation = self.bAnimation

    def OnResetButton(self, event):
        self.canvas.clothObject.resetMassSpring()

    def OnStiffnessSlider(self, event):
        val = event.GetEventObject().GetValue()
        stiffness = 2 ** val
        self.stiffnessLabel.SetLabel('Stiffness: ' + str(stiffness))
        self.canvas.clothObject.stiffness = stiffness

    def OnStepSlider(self, event):
        val = event.GetEventObject().GetValue()
        stepSize = 0.0001 * val
        self.stepLabel.SetLabel('Time Interval: ' + str(stepSize))
        self.canvas.stepSize = stepSize

    def OnDampSlider(self, event):
        val = event.GetEventObject().GetValue()
        damp = val / 10.0
        self.dampLabel.SetLabel('Damping Coeff.: ' + str(damp))
        self.canvas.clothObject.damp = damp


class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent):
        self.initialized = False
        self.size = (1024, 720)  # Canvas size
        self.aspect_ratio = 1
        self.stepSize = 0.0001
        glcanvas.GLCanvas.__init__(self, parent, -1, size=self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.Bind(wx.EVT_PAINT, self.OnDraw)  # Draw callback function
        self.Bind(wx.EVT_IDLE, self.OnIdle)  # Idle callback function
        self.InitGL()  # Initialization for OpenGL
        self.clothObject = ClothObject.ClothObject(1, 1, 10, 10)  # Create cloth object instance
        self.bAnimation = False  # Animation starts as False

    def InitGL(self):
        """Sets up the projection matrix and viewport"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.aspect_ratio = float(self.size[0]) / self.size[1]
        gluPerspective(60, self.aspect_ratio, 0.1, 100.0)
        glViewport(0, 0, self.size[0], self.size[1])

    def OnDraw(self, event):
        # Clear color and depth buffers
        if not self.initialized:
            self.InitGL()
            self.initialized = True
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clears canvas before drawing again

        # Position viewer
        glMatrixMode(GL_MODELVIEW)  # Chooses matrix to manipulate
        glLoadIdentity()
        gluLookAt(2,2,2, 0,0,0, 0,1,0)
        
        self.clothObject.drawSpring()

        self.SwapBuffers()

    def OnIdle(self, event):
        """Update state of mass-spring with dt"""
        if self.bAnimation:
            self.clothObject.update(self.stepSize)
        self.Refresh()


def main():
    app = wx.App()
    frame = MyFrame()
    frame.Show()
    app.MainLoop()


if __name__=='__main__':
    main()
