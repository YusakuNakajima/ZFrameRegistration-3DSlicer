import numpy as np
from scipy.fft import fft2, ifft2

class zf:
    @staticmethod
    def PrintMatrix(matrix):
        print("=============")
        for row in matrix:
            print(", ".join(str(x) for x in row))
        print("=============")
    
    @staticmethod
    def QuaternionToMatrix(q):
        """Convert quaternion to 4x4 matrix."""
        q = np.array(q)
        q = q / np.sqrt(np.sum(q * q))
        x, y, z, w = q

        xx = x * x * 2.0
        xy = x * y * 2.0
        xz = x * z * 2.0
        xw = x * w * 2.0
        yy = y * y * 2.0
        yz = y * z * 2.0
        yw = y * w * 2.0
        zz = z * z * 2.0
        zw = z * w * 2.0

        m = np.eye(4)
        m[0, 0] = 1.0 - (yy + zz)
        m[1, 0] = xy + zw
        m[2, 0] = xz - yw
        m[0, 1] = xy - zw
        m[1, 1] = 1.0 - (xx + zz)
        m[2, 1] = yz + xw
        m[0, 2] = xz + yw
        m[1, 2] = yz - xw
        m[2, 2] = 1.0 - (xx + yy)
        
        return m
    
    @staticmethod
    def MatrixToQuaternion(m):
        """Convert 4x4 matrix to quaternion."""
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        q = np.zeros(4)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (m[2, 1] - m[1, 2]) * s
            q[1] = (m[0, 2] - m[2, 0]) * s
            q[2] = (m[1, 0] - m[0, 1]) * s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
                q[3] = (m[2, 1] - m[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (m[0, 1] + m[1, 0]) / s
                q[2] = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
                q[3] = (m[0, 2] - m[2, 0]) / s
                q[0] = (m[0, 1] + m[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (m[1, 2] + m[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
                q[3] = (m[1, 0] - m[0, 1]) / s
                q[0] = (m[0, 2] + m[2, 0]) / s
                q[1] = (m[1, 2] + m[2, 1]) / s
                q[2] = 0.25 * s
                
        return q
    
    @staticmethod
    def Cross(a, b, c):
        a[0] = b[1]*c[2] - c[1]*b[2]
        a[1] = c[0]*b[2] - b[0]*c[2]
        a[2] = b[0]*c[1] - c[0]*b[1]
        
    @staticmethod
    def IdentityMatrix(matrix):
        matrix[0][0] = 1.0
        matrix[1][0] = 0.0
        matrix[2][0] = 0.0
        matrix[3][0] = 0.0
        matrix[0][1] = 0.0
        matrix[1][1] = 1.0
        matrix[2][1] = 0.0
        matrix[3][1] = 0.0

    @staticmethod
    def QuaternionMultiply(q1, q2):
        """Multiply two quaternions."""
        result = np.zeros(4)
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        result[0] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        result[1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        result[2] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        result[3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return result

    @staticmethod
    def QuaternionDivide(q1, q2):
        """Divide two quaternions (q1/q2 = q1 * inverse(q2))."""
        q1 = np.array(q1)
        q2 = np.array(q2)
        q2_norm = np.sum(q2 * q2)
        if q2_norm < 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0])
        q2_inv = np.array([-q2[0], -q2[1], -q2[2], q2[3]]) / q2_norm
        
        result = np.zeros(4)
        result[0] = q1[3]*q2_inv[0] + q1[0]*q2_inv[3] + q1[1]*q2_inv[2] - q1[2]*q2_inv[1]
        result[1] = q1[3]*q2_inv[1] - q1[0]*q2_inv[2] + q1[1]*q2_inv[3] + q1[2]*q2_inv[0]
        result[2] = q1[3]*q2_inv[2] + q1[0]*q2_inv[1] - q1[1]*q2_inv[0] + q1[2]*q2_inv[3]
        result[3] = q1[3]*q2_inv[3] - q1[0]*q2_inv[0] - q1[1]*q2_inv[1] - q1[2]*q2_inv[2]
        return result
    
    @staticmethod
    def QuaternionRotateVector(q, v):
        """Rotate a vector by a quaternion rotation."""
        qx, qy, qz, qw = q
        t = 2.0 * np.cross([qx, qy, qz], v)
        return v + qw * t + np.cross([qx, qy, qz], t)    
        
class ZFrameRegistration:
    def __init__(self, numFiducials=7):
        self.numFiducials = numFiducials
        self.InputImage = None
        self.InputImageDim = [0, 0, 0]
        self.InputImageTrans = None
        self.frameTopology = None
        self.manualRegistration = False
        self.zFrameFids = None
        self.ZOrientationBase = [0, 0, 0, 1]
        self.lastDetectedCoordinates = None
        # Modified: Initialize marker diameter
        self.markerDiameter = 11
        
        # Constants
        self.MEPSILON = 1e-10

    def getLastDetectedCoordinates(self):
        return self.lastDetectedCoordinates
    
    # Modified: Method to set marker diameter
    def SetMarkerDiameter(self, diameter):
        self.markerDiameter = int(diameter)
    
    def SetFrameTopology(self, frameTopology):
        self.frameTopology = frameTopology
    
    def SetInputImage(self, inputImage, transform):
        self.InputImage = inputImage.astype(int)
        self.InputImageDim = list(inputImage.shape)
        self.InputImageTrans = transform
        
    def SetOrientationBase(self, orientation):
        self.ZOrientationBase = orientation

    def Register(self, sliceRange):
        """Register Z-frame fiducials across multiple slices and compute average transformation."""
        xsize, ysize, zsize = self.InputImageDim
        
        tx = self.InputImageTrans[0][0]
        ty = self.InputImageTrans[1][0]
        tz = self.InputImageTrans[2][0]
        sx = self.InputImageTrans[0][1]
        sy = self.InputImageTrans[1][1]
        sz = self.InputImageTrans[2][1]
        nx = self.InputImageTrans[0][2]
        ny = self.InputImageTrans[1][2]
        nz = self.InputImageTrans[2][2]
        px = self.InputImageTrans[0][3]
        py = self.InputImageTrans[1][3]
        pz = self.InputImageTrans[2][3]
        
        psi = np.sqrt(tx*tx + ty*ty + tz*tz)
        psj = np.sqrt(sx*sx + sy*sy + sz*sz)
        psk = np.sqrt(nx*nx + ny*ny + nz*nz)
        ntx, nty, ntz = tx/psi, ty/psi, tz/psi
        nsx, nsy, nsz = sx/psj, sy/psj, sz/psj
        nnx, nny, nnz = nx/psk, ny/psk, nz/psk

        n = 0
        T = np.zeros((4, 4))
        P = np.zeros(3)
        all_detected_points = []

        matrix = np.eye(4)
        matrix[0:3, 0] = [ntx, nty, ntz]
        matrix[0:3, 1] = [nsx, nsy, nsz]
        matrix[0:3, 2] = [nnx, nny, nnz]
        
        print(f"Processing slices from {sliceRange[0]} to {sliceRange[1]}")
        for slindex in range(sliceRange[0], sliceRange[1]):
            print(f"=== Current Slice Index: {slindex} ===")
            hfovi = psi * (self.InputImageDim[0]-1) / 2.0
            hfovj = psj * (self.InputImageDim[1]-1) / 2.0
            offsetk = psk * slindex
            
            cx = ntx * hfovi + nsx * hfovj + nnx * offsetk
            cy = nty * hfovi + nsy * hfovj + nny * offsetk
            cz = ntz * hfovi + nsz * hfovj + nnz * offsetk
            
            quaternion = zf.MatrixToQuaternion(matrix)
            position = [px + cx, py + cy, pz + cz]
            
            if 0 <= slindex < zsize:
                current_slice = self.InputImage[:, :, slindex]
            else:
                return False, None, None, []
            
            self.Init(xsize, ysize)
            
            spacing = [psi, psj, psk]
            if self.RegisterQuaternion(position, quaternion, self.ZOrientationBase,
                                    current_slice, self.InputImageDim, spacing):
                P += np.array(position)
                
                q = np.array(quaternion)
                T += np.outer(q, q)
                n += 1
                
                if self.lastDetectedCoordinates:
                    all_detected_points.append({
                        "slice": slindex,
                        "points": self.lastDetectedCoordinates
                    })

            print(f"=== End Slice Index: {slindex} ===\n")
                
        if n <= 0:
            return False, None, None, []
            
        P /= float(n)
        T /= float(n)

        eigenvals, eigenvecs = np.linalg.eigh(T)
        max_idx = np.argmax(eigenvals)

        Zposition = P
        Zorientation = eigenvecs[:, max_idx]
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = zf.QuaternionToMatrix(Zorientation)[:3, :3]
        
        z_direction = transform_matrix[:3, 2]
        
        if np.dot(z_direction, np.array([0, 0, 1])) < 0:
            print("ZFrameRegistration - Correcting orientation to point superior")
            rot_matrix = np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            new_transform = np.dot(transform_matrix, rot_matrix)
            Zorientation = zf.MatrixToQuaternion(new_transform)
        
        return True, Zposition, Zorientation, all_detected_points

    # Modified: Dynamic kernel creation based on diameter
    def Init(self, xsize, ysize):
        """Initialize correlation kernel based on marker diameter."""
        # Create dynamic kernel based on diameter
        d = self.markerDiameter
        r = d / 2.0
        
        # Ensure kernel size is odd so it has a center pixel
        kernel_size = d if d % 2 == 1 else d + 1
        
        # Create grid
        y, x = np.ogrid[-kernel_size//2 + 1 : kernel_size//2 + 1, 
                        -kernel_size//2 + 1 : kernel_size//2 + 1]
        
        # Create disk mask (1.0 inside circle, 0.0 outside)
        mask = x*x + y*y <= r*r
        kernel = mask.astype(float)

        self.MaskImage = np.zeros((xsize, ysize))
        
        # Copy correlation kernel to center of mask image
        x_start = (xsize // 2) - (kernel_size // 2)
        y_start = (ysize // 2) - (kernel_size // 2)
        
        # Paste the kernel (handle boundary conditions if image is smaller than kernel, though unlikely)
        self.MaskImage[x_start:x_start+kernel_size, y_start:y_start+kernel_size] = kernel

        # Transform mask to frequency domain using FFT
        mask_fft = fft2(self.MaskImage)
        
        self.MFreal = np.real(mask_fft)
        self.MFimag = np.imag(mask_fft)
        
        self.MFimag *= -1
        max_absolute = np.max(np.abs(mask_fft))
        if max_absolute > 0:
            self.MFreal /= max_absolute
            self.MFimag /= max_absolute

    def RegisterQuaternion(self, position, quaternion, ZquaternionBase, SourceImage, dimension, spacing):
        """Register the Z-frame using quaternion representation."""
        Iposition = np.array(position)
        Iorientation = np.array(quaternion)
        ZorientationBase = np.array(ZquaternionBase)
        
        print("ZTrackerTransform - Searching fiducials...")
        Zcoordinates, tZcoordinates, center = self.LocateFiducials(SourceImage, dimension[0], dimension[1])
        if Zcoordinates is None:
            print("ZTrackerTransform::onEventGenerated - Fiducials not detected. No frame lock on this image.")
            return False
        
        self.lastDetectedCoordinates = Zcoordinates
        
        print("ZTrackerTransform - Checking the fiducial geometries...")
        print(f"Zcoordinates: {Zcoordinates}")
        if not self.CheckFiducialGeometry(Zcoordinates, dimension[0], dimension[1]):
            print("ZTrackerTransform::onEventGenerated - Bad fiducial geometry. No frame lock on this image.")
            return False
        
        for i in range(self.numFiducials):
            tZcoordinates[i][0] = float(tZcoordinates[i][0]) - center[0]
            tZcoordinates[i][1] = float(tZcoordinates[i][1]) - center[1]
            
            tZcoordinates[i][0] *= spacing[0]
            tZcoordinates[i][1] *= spacing[1]
        
        Zposition, Zorientation = self.LocalizeFrame(tZcoordinates)
        if Zposition is None or Zorientation is None:
            print("ZTrackerTransform::onEventGenerated - Could not localize the frame. Skipping this one.")
            return False
        
        rotated_position = zf.QuaternionRotateVector(Iorientation, Zposition)
        Zposition = Iposition + rotated_position
        
        Zorientation = zf.QuaternionMultiply(Iorientation, Zorientation)
        Zorientation = zf.QuaternionDivide(Zorientation, ZorientationBase)
        
        position[0] = Zposition[0]
        position[1] = Zposition[1]
        position[2] = Zposition[2]
        quaternion[0] = Zorientation[0]
        quaternion[1] = Zorientation[1]
        quaternion[2] = Zorientation[2]
        quaternion[3] = Zorientation[3]
        
        return True

    def LocateFiducials(self, SourceImage, xsize, ysize):
        """Locate the fiducial intercepts in the Z-frame."""
        Zcoordinates = [[0, 0] for _ in range(self.numFiducials)]
        tZcoordinates = [[0.0, 0.0] for _ in range(self.numFiducials)]
        
        image_fft = fft2(SourceImage)
        IFreal = np.real(image_fft)
        IFimag = np.imag(image_fft)
        
        max_absolute = np.max(np.abs(image_fft))
        if max_absolute < self.MEPSILON:
            print("ZTrackerTransform::LocateFiducials - divide by zero.")
            return None, None, None
            
        IFreal /= max_absolute
        IFimag /= max_absolute
        
        PFreal = IFreal * self.MFreal - IFimag * self.MFimag
        PFimag = IFreal * self.MFimag + IFimag * self.MFreal
        
        product_ifft = ifft2(PFreal + 1j * PFimag)
        PIreal = np.real(product_ifft)
        PIreal = np.fft.fftshift(PIreal)
        
        max_absolute = np.max(np.abs(PIreal))
        if max_absolute < self.MEPSILON:
            print("ZTrackerTransform::LocateFiducials - divide by zero.")
            return None, None, None
            
        PIreal /= max_absolute
        
        peak_count = 0
        for i in range(self.numFiducials):
            peak_val, peak_coords = self.FindMax(PIreal)
            Zcoordinates[i] = list(peak_coords)
            
            # Mask out region based on diameter to avoid finding the same peak
            d = self.markerDiameter
            margin = d # Use diameter as margin
            
            rstart = max(0, peak_coords[0] - margin)
            rstop = min(xsize - 1, peak_coords[0] + margin)
            cstart = max(0, peak_coords[1] - margin)
            cstop = min(ysize - 1, peak_coords[1] + margin)
            
            if peak_val < self.MEPSILON:
                print("Registration::OrderFidPoints - peak value is zero.")
                return None, None, None
            
            # Simplified peak check
            # For large markers, the standard offpeak check might be too sensitive
            # Just proceeding if peak is strong enough relative to global max (which is 1.0)
            if peak_val < 0.3:
                 i -= 1
                 print(f"Registration::LocateFiducials - Peak too weak ({peak_val}).")
                 peak_count += 1
                 if peak_count > 20: # Allow more retries
                     return None, None, None
                 continue

            tZcoordinates[i] = self.FindSubPixelPeak(
                peak_coords,
                peak_val,
                PIreal[peak_coords[0]-1, peak_coords[1]],
                PIreal[peak_coords[0]+1, peak_coords[1]],
                PIreal[peak_coords[0], peak_coords[1]-1],
                PIreal[peak_coords[0], peak_coords[1]+1]
            )
            
            PIreal[rstart:rstop+1, cstart:cstop+1] = 0.0
        
        center = self.FindFidCentre(tZcoordinates)
        self.FindFidCorners(tZcoordinates, center)
        self.OrderFidPoints(tZcoordinates, center[0], center[1])
        
        for i in range(self.numFiducials):
            Zcoordinates[i] = [int(tZcoordinates[i][0]), int(tZcoordinates[i][1])]
        
        return Zcoordinates, tZcoordinates, center

    def FindSubPixelPeak(self, peak_coords, Y0, Yx1, Yx2, Yy1, Yy2):
        Xshift = (0.5 * (Yx1 - Yx2)) / (Yx1 + Yx2 - 2.0 * Y0)
        Yshift = (0.5 * (Yy1 - Yy2)) / (Yy1 + Yy2 - 2.0 * Y0)
        
        if abs(Xshift) > 1.0 or abs(Yshift) > 1.0:
            return [float(peak_coords[0]), float(peak_coords[1])]
        
        return [float(peak_coords[0]) + Xshift, float(peak_coords[1]) + Yshift]

    def CheckFiducialGeometry(self, Zcoordinates, xsize, ysize):
        print("Registration::CheckFiducialGeometry - Checking fiducial geometry...")
        for coord in Zcoordinates:
            if coord[0] < 0 or coord[0] >= xsize or coord[1] < 0 or coord[1] >= ysize:
                return False

        def get_normalized_vector(p1, p2):
            vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            norm = np.linalg.norm(vector)
            return vector / norm if norm != self.MEPSILON else vector

        if self.numFiducials == 7:
            P1 = np.array(Zcoordinates[0])
            P3 = np.array(Zcoordinates[2])
            P5 = np.array(Zcoordinates[4])
            P7 = np.array(Zcoordinates[6])

            D71 = get_normalized_vector(P1, P7)
            D53 = get_normalized_vector(P3, P5)
            D13 = get_normalized_vector(P3, P1)
            D75 = get_normalized_vector(P5, P7)
            
            dotp = abs(np.dot(D71, D53))
            if dotp < np.cos(5.0 * np.pi / 180.0):
                return False

            dotp = abs(np.dot(D13, D75))
            if dotp < np.cos(5.0 * np.pi / 180.0):
                return False
        elif self.numFiducials == 9:
            return True
        else:
            return False

        return True

    def FindFidCentre(self, points):
        minrow = maxrow = points[0][0]
        mincol = maxcol = points[0][1]
        
        for point in points:
            minrow = min(minrow, point[0])
            maxrow = max(maxrow, point[0])
            mincol = min(mincol, point[1])
            maxcol = max(maxcol, point[1])
        
        rmid = (minrow + maxrow)/2.0
        cmid = (mincol + maxcol)/2.0
        return rmid, cmid

    def FindFidCorners(self, points, pmid):
        distances = [self.CoordDistance(point, pmid) for point in points]
        
        swapped = True
        while swapped:
            swapped = False
            for i in range(6): 
                if distances[i] < distances[i+1]:
                    distances[i], distances[i+1] = distances[i+1], distances[i]
                    points[i], points[i+1] = points[i+1], points[i]
                    swapped = True
        
        pdist1 = self.CoordDistance(points[0], points[1])
        pdist2 = self.CoordDistance(points[0], points[2])
        if pdist1 > pdist2:
            points[1], points[2] = points[2], points[1]
        
        pdist1 = self.CoordDistance(points[1], points[2])
        pdist2 = self.CoordDistance(points[1], points[3])
        if pdist1 > pdist2:
            points[2], points[3] = points[3], points[2]

    def CoordDistance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx*dx + dy*dy)

    def OrderFidPoints(self, points, rmid, cmid):
        if self.numFiducials == 7:
            pall = [0, -1, 1, -1, 2, -1, 3, -1, 0]
            pother = [4, 5, 6]
                
            for i in range(0, 7, 2):
                for j in range(3):
                    if pother[j] == -1: continue
                        
                    cdist = self.CoordDistance(points[pall[i]], points[pall[i+2]])
                    pdist1 = self.CoordDistance(points[pall[i]], points[pother[j]])
                    pdist2 = self.CoordDistance(points[pall[i+2]], points[pother[j]])
                    
                    if cdist < self.MEPSILON: continue
                        
                    if ((pdist1 + pdist2) / cdist) < 1.05:
                        pall[i+1] = pother[j]
                        pother[j] = -1
                        break
            
            for i in range(1, 9):
                if pall[i] == -1: break
            
            d1x = points[pall[0]][0] - rmid
            d1y = points[pall[0]][1] - cmid
            d2x = points[pall[2]][0] - rmid
            d2y = points[pall[2]][1] - cmid
            nvecz = (d1x * d2y - d2x * d1y)
            
            direction = -1 if nvecz < 0 else 1
            
            pall2 = []
            curr_i = i
            for _ in range(7):
                curr_i += direction
                if curr_i == -1: curr_i = 7
                if curr_i == 9: curr_i = 1
                pall2.append(pall[curr_i])
            
            points_temp = [[points[idx][0], points[idx][1]] for idx in pall2]
            for i in range(7):
                points[i][0] = points_temp[i][0]
                points[i][1] = points_temp[i][1]
        elif self.numFiducials == 9:
            sorter_array = [9, 1, 4, 6, 0, 0, 0, 0, 0]
            sorted_points = [True, True, True, True, False, False, False, False, False]
            
            shortest_dist = 10000
            closest_index = 0
            for i in range(4, 9):
                coord_dist = self.CoordDistance(points[1], points[i])
                if coord_dist < shortest_dist:
                    shortest_dist = coord_dist
                    closest_index = i
            sorter_array[closest_index] = 2
            sorted_points[closest_index] = True
            
            shortest_dist = 10000
            for i in range(4, 9):
                if not sorted_points[i]:
                    coord_dist = self.CoordDistance(points[1], points[i])
                    if coord_dist < shortest_dist:
                        shortest_dist = coord_dist
                        closest_index = i
            sorter_array[closest_index] = 3
            sorted_points[closest_index] = True
            
            shortest_dist = 10000
            for i in range(4, 9):
                if not sorted_points[i]:
                    coord_dist = self.CoordDistance(points[2], points[i])
                    if coord_dist < shortest_dist:
                        shortest_dist = coord_dist
                        closest_index = i
            sorter_array[closest_index] = 5
            sorted_points[closest_index] = True
            
            shortest_dist = 10000
            for i in range(4, 9):
                if not sorted_points[i]:
                    coord_dist = self.CoordDistance(points[0], points[i])
                    if coord_dist < shortest_dist:
                        shortest_dist = coord_dist
                        closest_index = i
            sorter_array[closest_index] = 8
            sorted_points[closest_index] = True
            
            for i in range(4, 9):
                if not sorted_points[i]:
                    sorter_array[i] = 7
                    sorted_points[i] = True
            
            pairs = [(sorter_array[i], [points[i][0], points[i][1]]) 
                    for i in range(9)]
            pairs.sort(key=lambda x: x[0])
            
            for i in range(9):
                points[i][0] = pairs[i][1][0]
                points[i][1] = pairs[i][1][1]
            
            if points[0][0] > points[8][0]:
                points.reverse()

    def LocalizeFrame(self, Zcoordinates):
        def make_vector(x, y, z=0.0):
            return np.array([x, y, z])
        
        if self.numFiducials == 7:
            Pz1 = make_vector(Zcoordinates[0][0], Zcoordinates[0][1])
            Pz2 = make_vector(Zcoordinates[1][0], Zcoordinates[1][1])
            Pz3 = make_vector(Zcoordinates[2][0], Zcoordinates[2][1])
            Oz = make_vector(*self.frameTopology[0])
            Vz = make_vector(*self.frameTopology[3])
            fiducialDistance = np.abs(self.frameTopology[0][1]*2)
            P2f = self.SolveZ(Pz1, Pz2, Pz3, Oz, Vz, fiducialDistance)
            if P2f is None: return None, None
                
            Pz1 = make_vector(Zcoordinates[2][0], Zcoordinates[2][1])
            Pz2 = make_vector(Zcoordinates[3][0], Zcoordinates[3][1])
            Pz3 = make_vector(Zcoordinates[4][0], Zcoordinates[4][1])
            Oz = make_vector(*self.frameTopology[1])
            Vz = make_vector(*self.frameTopology[4])
            fiducialDistance = np.abs(self.frameTopology[1][0]*2)
            P4f = self.SolveZ(Pz1, Pz2, Pz3, Oz, Vz, fiducialDistance)
            if P4f is None: return None, None
                
            Pz1 = make_vector(Zcoordinates[4][0], Zcoordinates[4][1])
            Pz2 = make_vector(Zcoordinates[5][0], Zcoordinates[5][1])
            Pz3 = make_vector(Zcoordinates[6][0], Zcoordinates[6][1])
            Oz = make_vector(*self.frameTopology[2])
            Vz = make_vector(*self.frameTopology[5])
            fiducialDistance = np.abs(self.frameTopology[2][1]*2)
            P6f = self.SolveZ(Pz1, Pz2, Pz3, Oz, Vz, fiducialDistance)
            if P6f is None: return None, None
            
            Vx = P2f - P6f
            Vy = P4f - P6f
            Vx = Vx / np.linalg.norm(Vx)
            Vz = np.cross(Vx, Vy)
            Vz = Vz / np.linalg.norm(Vz)
            Vy = np.cross(Vz, Vx)
            
            rotation_matrix = np.column_stack((Vx, Vy, Vz))
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            Qft = zf.MatrixToQuaternion(transform_matrix) 
            
            Pz1 = make_vector(Zcoordinates[1][0], Zcoordinates[1][1])
            Pz2 = make_vector(Zcoordinates[3][0], Zcoordinates[3][1])
            Pz3 = make_vector(Zcoordinates[5][0], Zcoordinates[5][1])
            Vx = Pz1 - Pz3
            Vy = Pz2 - Pz3
            Vz = np.cross(Vx, Vy)
            Vy = np.cross(Vz, Vx)

            Vx_norm = np.linalg.norm(Vx)
            Vy_norm = np.linalg.norm(Vy)
            Vz_norm = np.linalg.norm(Vz)

            if Vx_norm < self.MEPSILON or Vy_norm < self.MEPSILON or Vz_norm < self.MEPSILON:
                return None, None

            Vx = Vx / Vx_norm
            Vy = Vy / Vy_norm
            Vz = Vz / Vz_norm
            
            rotation_matrix = np.column_stack((Vx, Vy, Vz))
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            Qit = zf.MatrixToQuaternion(transform_matrix)
            
            Zorientation = zf.QuaternionDivide(Qit, Qft)
            
            angle = 2 * np.arccos(Zorientation[3])
            if abs(angle) > 50.0:
                return None, None
            
            Cf = (P2f + P4f + P6f) / 3.0
            Cfi = zf.QuaternionRotateVector(Zorientation, Cf)
            Ci = (Pz1 + Pz2 + Pz3) / 3.0
            Zposition = Ci - Cfi
            
            if abs(Zposition[2]) > 100.0:
                return None, None
            
            return Zposition, Zorientation
        elif self.numFiducials == 9:
            Pz1 = make_vector(Zcoordinates[0][0], Zcoordinates[0][1])
            Pz2 = make_vector(Zcoordinates[1][0], Zcoordinates[1][1])
            Pz3 = make_vector(Zcoordinates[2][0], Zcoordinates[2][1])
            Oz = make_vector(*self.frameTopology[0])
            Vz = make_vector(*self.frameTopology[3])
            fiducialDistance = np.abs(self.frameTopology[0][1]*2)
            P2f = self.SolveZ(Pz1, Pz2, Pz3, Oz, Vz, fiducialDistance)
            if P2f is None: return None, None
            
            Pz1 = make_vector(Zcoordinates[3][0], Zcoordinates[3][1])
            Pz2 = make_vector(Zcoordinates[4][0], Zcoordinates[4][1])
            Pz3 = make_vector(Zcoordinates[5][0], Zcoordinates[5][1])
            Oz = make_vector(*self.frameTopology[1])
            Vz = make_vector(*self.frameTopology[4])
            fiducialDistance = np.abs(self.frameTopology[1][0]*2)
            P4f = self.SolveZ(Pz1, Pz2, Pz3, Oz, Vz, fiducialDistance)
            if P4f is None: return None, None
            
            Pz1 = make_vector(Zcoordinates[6][0], Zcoordinates[6][1])
            Pz2 = make_vector(Zcoordinates[7][0], Zcoordinates[7][1])
            Pz3 = make_vector(Zcoordinates[8][0], Zcoordinates[8][1])
            Oz = make_vector(*self.frameTopology[2])
            Vz = make_vector(*self.frameTopology[5])
            fiducialDistance = np.abs(self.frameTopology[2][1]*2)
            P6f = self.SolveZ(Pz1, Pz2, Pz3, Oz, Vz, fiducialDistance)
            if P6f is None: return None, None
            
            Vx = P2f - P6f
            Vy = P4f - P6f
            Vx_norm = np.linalg.norm(Vx)
            if Vx_norm < self.MEPSILON: return None, None
            Vx = Vx / Vx_norm
            Vz = np.cross(Vx, Vy)
            Vz_norm = np.linalg.norm(Vz)
            if Vz_norm < self.MEPSILON: return None, None
            Vz = Vz / Vz_norm
            Vy = np.cross(Vz, Vx)
            
            rotation_matrix = np.column_stack((Vx, Vy, Vz))
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            Qft = zf.MatrixToQuaternion(transform_matrix)
            
            if abs(Zcoordinates[4][0]) > 10:
                return None, None
            
            Pz1 = make_vector(Zcoordinates[1][0], Zcoordinates[1][1])
            Pz2 = make_vector(Zcoordinates[4][0], Zcoordinates[4][1])
            Pz3 = make_vector(Zcoordinates[7][0], Zcoordinates[7][1])
            Vx = Pz1 - Pz3
            Vy = Pz2 - Pz3
            Vz = np.cross(Vx, Vy)
            Vy = np.cross(Vz, Vx)

            Vx_norm = np.linalg.norm(Vx)
            Vy_norm = np.linalg.norm(Vy)
            Vz_norm = np.linalg.norm(Vz)

            if Vx_norm < self.MEPSILON or Vy_norm < self.MEPSILON or Vz_norm < self.MEPSILON:
                return None, None
            
            Vx = Vx / Vx_norm
            Vy = Vy / Vy_norm
            Vz = Vz / Vz_norm
            
            rotation_matrix = np.column_stack((Vx, Vy, Vz))
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            Qit = zf.MatrixToQuaternion(transform_matrix)
            
            Zorientation = zf.QuaternionDivide(Qit, Qft)
            angle = 2 * np.arccos(Zorientation[3])
            if abs(angle) > 50.0:
                return None, None
            
            Cf = (P2f + P4f + P6f) / 3.0
            Cfi = zf.QuaternionRotateVector(Zorientation, Cf)
            Ci = (Pz1 + Pz2 + Pz3) / 3.0
            Zposition = Ci - Cfi
            if abs(Zposition[2]) > 100.0:
                return None, None
            
            return Zposition, Zorientation

    def SolveZ(self, P1, P2, P3, Oz, Vz, fiducialDistance):
        try:
            diagonalLength = np.linalg.norm(Vz)
            Vz_normalized = Vz / diagonalLength
            D12 = np.linalg.norm(P1 - P2)
            D23 = np.linalg.norm(P2 - P3)
            if D12 + D23 < self.MEPSILON: return None
            Lc = diagonalLength * D23 / (D12 + D23)
            P2f = Oz + Vz_normalized * Lc
            return P2f
        except Exception:
            return None
        
    def FindMax(self, matrix):
        rows, cols = matrix.shape
        max_val = 0
        max_coords = [0, 0]
        # Avoid margin
        for i in range(10, rows-10):
            for j in range(10, cols-10):
                if max_val < matrix[i, j]:
                    max_val = matrix[i, j]
                    max_coords = [i, j]
        return max_val, max_coords