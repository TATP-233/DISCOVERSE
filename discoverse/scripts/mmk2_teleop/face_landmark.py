import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    if not len(face_landmarks_list):
        return rgb_image

    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ]
    )

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
    )
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
    )

    return annotated_image


class HeadLandmarks:
    def __init__(self) -> None:
        base_options = python.BaseOptions(
            model_asset_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "face_landmarker_v2_with_blendshapes.task",
            )
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            min_face_detection_confidence=0.02,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def proc(self, img_cv_bgr):
        # STEP 3: Load the input image.
        img_RGB = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_RGB)

        # STEP 4: Detect face landmarks from the input image.
        detection_result = self.detector.detect(mp_image)
        return detection_result


if __name__ == "__main__":

    hl = HeadLandmarks()

    cv2.namedWindow("Face Landmarks")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection_result = hl.proc(frame)
        annotated_image = draw_landmarks_on_image(frame, detection_result)

        cv2.imshow("Face Landmarks", annotated_image)
        # cv2.imshow("Face Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyWindow("Face Landmarks")
