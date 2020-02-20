import numpy
import math

from OpenGL.GL import *
from OpenGL.GLU import *


def convertLongVector(x):
    """Concatenate 3D vectors in array x to create large vector longX"""
    longX = numpy.zeros(shape=(len(x)*3,))
    for i in range(0, len(x)):
        longX[i*3 : i*3+3] = x[i]
    return longX


def set33SubMatrix(M, i, j, m):
    """
    Set submatrix of large matrix M at i-th row and j-th column with
    small 3x3 matrix m
    """
    for row in range(0, 3):
        for col in range(0, 3):
            M[i*3+row, j*3+col] = m[row, col]


def set33SubMatrixSymmetric(M, i, j, m):
    set33SubMatrix(M, i, j, m)
    set33SubMatrix(M, j, i, m)
    

class ClothObject:

    def __init__(self, sizeX, sizeZ, nW, nH):

        # Relaxed spring lengths for x, y, diagonal
        self.dx = sizeX / (nW - 1)
        self.dz = sizeZ / (nH - 1)
        self.dDiag = math.sqrt(self.dx**2.0 + self.dz**2.0)

        self.damp = 0.0

        self.nW, self.nH = nW, nH
        self.clothSize = (sizeX, sizeZ)

        # Vertex array (x, y, z)
        self.verts = numpy.zeros(shape=(nW * nH, 3))
        self.velocity = numpy.zeros(shape=(nW * nH, 3))

        # Spring links (i, j)
        self.springs = numpy.zeros(shape=((nW - 1) * nH + (nH - 1) * nW + (nW - 1) * (nH - 1) * 2, 2),
                                   dtype=numpy.int32)

        # Relaxed length of spring: l0
        self.l0 = numpy.zeros(shape=(len(self.springs),))

        self.numXSprings = (self.nW - 1) * self.nH
        self.numZSprings = (self.nH - 1) * self.nW
        self.numDiagonalSprings = (self.nW - 1) * (self.nH - 1)

        # Force array for fi
        self.force = numpy.zeros(shape=(nW * nH, 3))

        self.totalMass = 1.0
        self.stiffness = 1.0  # Stiffness K for all springs
        self.totalNumberOfParticles = nW * nH

        self.mass = numpy.zeros(shape=(len(self.verts),))  # Mass array for mi
        particleMass = self.totalMass / self.totalNumberOfParticles
        for i in range(0, self.totalNumberOfParticles):
            self.mass[i] = particleMass

        # Data for force derivative
        self.J = numpy.zeros(shape=(self.totalNumberOfParticles*3, self.totalNumberOfParticles*3))

        self.resetMassSpring()


    def resetMassSpring(self):
        """Initializes location of vertices, sets up spring links, computes relaxed length"""
        
        # Initialize location and velocity of vertices
        for i in range(0, self.nW * self.nH):
            self.verts[i] = [(i % self.nW) * self.dx, 1, (int(i / self.nW)) * self.dz]
            self.velocity[i] = [0, 0, 0]

        for i in range(0, self.numXSprings):
            row = int(i / (self.nW - 1))
            col = i % (self.nW - 1)
            self.springs[i] = [row * self.nW + col, row * self.nW + col + 1]
            self.l0[i] = self.dx

        for i in range(0, self.numZSprings):
            row = int(i / (self.nW))
            col = i % (self.nW)
            self.springs[self.numXSprings + i] = [row * self.nW + col, (row + 1) * self.nW + col]
            self.l0[(self.nW - 1) * self.nH + i] = self.dz

        for i in range(0, self.numDiagonalSprings):
            row = int(i / (self.nW - 1))
            col = i % (self.nW - 1)
            self.springs[self.numXSprings + self.numZSprings + 2 * i] = \
                [row * self.nW + col, (row + 1) * self.nW + col + 1]
            self.l0[self.numXSprings + self.numZSprings + 2 * i] = self.dDiag

            self.springs[self.numXSprings + self.numZSprings + 2 * i + 1] = \
                [row * self.nW + col + 1, (row + 1) * self.nW + col]
            self.l0[self.numXSprings + self.numZSprings + 2 * i + 1] = self.dDiag

        
    def drawSpring(self):
        """Drawing method"""
        glColor3f(1, 1, 1)

        glBegin(GL_LINES)
        for i in range(0, len(self.springs)):
            # Get indices of linked vertices
            idx0 = self.springs[i][0]
            idx1 = self.springs[i][1]
            # Get vertices from indices
            loc0 = self.verts[idx0]
            loc1 = self.verts[idx1]
            # Connect vertices with a line
            glVertex3fv(loc0)
            glVertex3fv(loc1)
        glEnd()
    

    def computeForce(self):
        # Reset forces
        for i in range(0, self.totalNumberOfParticles):
            self.force[i] = numpy.array([0.0, 0.0, 0.0])  # Resets force array
        # Compute spring forces
        for i in range(0, len(self.springs)):
            idx0 = self.springs[i][0]
            idx1 = self.springs[i][1]
            xi = self.verts[idx0]
            xj = self.verts[idx1]
            xji = xj - xi
            vi = self.velocity[idx0]
            vj = self.velocity[idx1]
            vji = vj - vi
            l = numpy.linalg.norm(xji)
            dir = xji / l
            deform = l - self.l0[i]
            f = self.stiffness * deform * dir + self.damp * vji
            self.force[idx0] = self.force[idx0] + f
            self.force[idx1] = self.force[idx1] - f


    def integrateImplicit(self, dt):
        Jii = numpy.zeros(shape=(len(self.verts), 3, 3))
        bigM = numpy.zeros(shape=(len(self.verts)*3, len(self.verts)*3))

        for i in range(0, len(self.verts)):
            set33SubMatrix(bigM, i, i, self.mass[i]*numpy.identity(3))

        # Compute force derivative
        for e in range(0, len(self.springs)):
            i, j = self.springs[e][0], self.springs[e][1]
            xji = self.verts[j] - self.verts[i]
            lt = numpy.linalg.norm(xji)
            Jij = self.stiffness * (((lt - self.l0[e]) / lt) * numpy.identity(3) +
                                      (self.l0[e] / (lt ** 3.0)) * numpy.outer(xji, xji))
            Jii[i], Jii[j] = Jii[i] - Jij, Jii[j] - Jij
            set33SubMatrixSymmetric(self.J, i, j, Jij)

        for i in range(0, len(self.verts)):
            set33SubMatrix(self.J, i, i, Jii[i])

        # Compute velocity change with implicit method
        bigM = bigM - dt * dt* self.J
        bigMinv = numpy.linalg.inv(bigM)
        longV = convertLongVector(self.velocity)
        longF = convertLongVector(self.force)

        return bigMinv.dot(longF * dt + (dt ** 2) * self.J.dot(longV))


    def update(self, dt):
        """Implements falling vertices with gravity"""
        gravity = numpy.array([0.0, -9.8, 0.0])
        self.computeForce()  # Spring forces must be computed

        dVel = self.integrateImplicit(dt)  # Implicit integration to get velocity changes

        for i in range(0, self.totalNumberOfParticles):
            if i is not 0 and i is not self.nW - 1:  # These 2 masses are static
                # dt = h = integration step = time intervals the cloth is drawn at
                self.velocity[i] = self.velocity[i] + dVel[i*3 : i*3+3] + gravity * dt
                self.verts[i] = self.verts[i] + self.velocity[i] * dt
