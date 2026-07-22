import cv2
from lxml import etree

from TSR import table_structure_recognition_all as tsra
from TSR import table_structure_recognition_lines as tsrl
from TSR import table_structure_recognition_lines_wol as tsrlwol
from TSR import table_structure_recognition_wol as tsrwol


class TSR:
    _MODELS = {
        "bordered": tsrl,
        "unbordered": tsrwol,
        "partially": tsrlwol,
        "partially_color_inv": tsra,
    }

    def __init__(self, table_type="partially"):
        if table_type not in self._MODELS:
            raise ValueError(
                f"Unknown table type '{table_type}'. "
                f"Choose from {list(self._MODELS.keys())}"
            )

        self.model = self._MODELS[table_type]

    def _boxes_to_xml(self, table_boxes):
        root = etree.Element("table")

        for row_idx, row in enumerate(table_boxes):
            for col_idx, cell_boxes in enumerate(row):
                if not cell_boxes:
                    continue

                x, y, w, h = cell_boxes[0]

                cell = etree.SubElement(
                    root,
                    "cell",
                    row=str(row_idx),
                    column=str(col_idx),
                )

                etree.SubElement(
                    cell,
                    "boundingbox",
                    x=str(x),
                    y=str(y),
                    w=str(w),
                    h=str(h),
                )

        return etree.tostring(
            root,
            pretty_print=True,
            encoding="unicode",
        )

    def predict(self, image):
        """
        Parameters
        ----------
        image : np.ndarray
            OpenCV image (BGR)

        Returns
        -------
        xml : str
            XML representation of the table.
        processed_image : np.ndarray
            Image with detected structure.
        """
        boxes, processed_image = self.model.recognize_structure(image)
        xml = self._boxes_to_xml(boxes)

        return xml, processed_image

    def predict_from_path(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.predict(image)