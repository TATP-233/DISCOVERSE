import cv2
import numpy as np


class ArucoDetector:
    def __init__(self, aruco_dict, camera_matrix, dist_coeffs, marker_size):
        aruco_dict_pre = cv2.aruco.getPredefinedDictionary(aruco_dict)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict_pre, parameters)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size

    def estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        """
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        """
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        rvecs = []
        Rmats = []
        tvecs = []
        for c in corners:
            _, r, t = cv2.solvePnP(
                marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            R, _ = cv2.Rodrigues(r)
            rvecs.append(r)
            Rmats.append(R)
            tvecs.append(t)
        return rvecs, Rmats, tvecs

    def draw_aruco_axis(self, image, corners_int, ids, rvecs, tvecs):
        cv2.drawContours(image, corners_int, -1, (0, 255, 0), 3)
        # Draw the marker axes on the image
        for i in range(len(rvecs)):
            cv2.drawFrameAxes(
                image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.02
            )
            cv2.putText(
                image,
                str(ids[i][0]),
                tuple(corners_int[i][0][0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return image

    def detect(self, image, putAxis=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if len(corners) > 0:
            # Estimate the pose of each marker
            rvecs, Rmats, tvecs = self.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            if putAxis:
                corners_int = np.array(corners).astype(np.int32)
                self.draw_aruco_axis(image, corners_int, ids, rvecs, tvecs)
            return image, Rmats, rvecs, tvecs, ids, corners
        else:
            return image, [], [], [], ids, corners


if __name__ == "__main__":

    img_file_path = "<path_to_your_image>/xxx.png"
    img_file_path = "/home/tatp/Desktop/look_down/color_000.png"
    img = cv2.imread(img_file_path)
    print(img.shape)

    fx = 608.39551
    fy = 608.58689
    cx = 316.7065
    cy = 260.29088

    k1 = 0.08828
    k2 = -0.00769
    p1 = 0.00005
    p2 = 0.00059
    k3 = -0.56199

    camera_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera_distCoeffs = np.array([k1, k2, p1, p2, k3])
    marker_size = 0.03  # meter

    det = ArucoDetector(
        cv2.aruco.DICT_4X4_50, camera_intrinsic, camera_distCoeffs, marker_size
    )

    img, Rmats, rvecs, tvecs, ids, _ = det.detect(img)
    print(Rmats)  # 旋转
    print(tvecs)  # 平移
    print(ids)  # aruco id

    cv2.namedWindow("img")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
