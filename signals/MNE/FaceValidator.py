class FaceValidator:

    # face IDs based on eperimental setup in
    @staticmethod
    def _is_angry(face_id):
        labels = {1, 4, 7, 10, 13, 16, 20, 22}
        for i in labels:
            if face_id == 33024 + i:
                return True
        return False

    @staticmethod
    def _is_happy(face_id):
        labels = {3, 5, 8, 12, 14, 18, 21, 23}
        for i in labels:
            if face_id == 33024 + i:
                return True
        return False

    @staticmethod
    def is_emotional(face_id):
        if FaceValidator._is_angry(face_id) or FaceValidator._is_happy(face_id):
            return True
        return False
