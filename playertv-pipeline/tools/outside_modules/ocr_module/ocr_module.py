from paddleocr import PaddleOCR
import easyocr


class OCR_Module:
    def __init__(self, ocr_parameters, frame_rate = 30):
        self.parameters = ocr_parameters
        self.parameters["sample_rate"] = frame_rate
        self.frame_rate = frame_rate
        self.results = None

    def set_results(self, results):
         self.results = results

    def get_ocr_parameters(self):
        return self.parameters 
    
    
    def calculate_score(self, txt, conf):
        accumulator = 1
        conf = float(conf)
        try:
            txt = int(txt)
            accumulator = accumulator*2
        except:
            return (conf*accumulator, txt)
        return (conf*accumulator, txt)


    def map_id_to_kit(self):
        mapping = {}
        for key, val in self.results.items():
            top_score = 0
            kit_number = None
            for i in range(len(val["txt"])):
                for j in range(len(val["txt"][i])):
                    conf = val["conf"][i][j]
                    txt = val["txt"][i][j]
                    score, txt = self.calculate_score(txt, conf)
                    if score > top_score:
                        kit_number = txt
            mapping[key] = kit_number
        #Dict with key id, and value proposed kit number
        return mapping

    def run_ocr(self, image):
        raise NotImplementedError("Subclasses must implement run_ocr method.")


class EasyOCRModule(OCR_Module):
    def __init__(self, ocr_parameters, frame_rate = 30):
        self.model = easyocr.Reader(**ocr_parameters)
        super().__init__(ocr_parameters, frame_rate)

    def run_ocr(self, image):
        results = self.model.readtext(image, allowlist ='0123456789')
        if results:
            boxes = []
            txts = []
            conf = []
            for line in results:
                try:
                    integer = int(line[1])
                    if 0 < integer < 100:
                        txts.append(integer)
                        boxes.append(line[0])
                        conf.append(line[2])
                    else:
                        raise Exception("Integer not within range 1 - 99, skipping detection")
                except:
                    pass

            return txts, boxes, conf
        return [], [], []




class PaddleOCRModule(OCR_Module):
    def __init__(self, ocr_parameters, frame_rate = 30):
        self.model = PaddleOCR(**ocr_parameters)
        super().__init__(ocr_parameters, frame_rate)

    def run_ocr(self, image):
        result = self.model.ocr(image, cls = True)
        result = result[0]
        if result:
            boxes = []
            txts = []
            scores = []
            for line in result:
                try:
                    integer = int(line[1][0])
                    if 0 < integer < 100:
                        txts.append(integer)
                        boxes.append(line[0])
                        scores.append(line[1][1])
                    else:
                        raise Exception("Integer not within range 1 - 99, skipping detection")
                except:
                    pass
            return txts, boxes, scores
        return [], [], []