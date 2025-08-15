import cv2
import torch
import numpy as np
from torchvision import models, transforms
import time
import sys

# ---------- Configurações ----------
VIDEO_PATH = "videos/Vídeo 1 - 14_06_2024 - 16h até 16h20.mp4"
RESIZE_FACTOR = 0.5
SKIP_FRAMES = 2
VEHICLE_CLASSES = [7, 13]  # IDs de caminhão e carro
CLASS_COLORS = {
    0: [0, 0, 0],  # Fundo
    7: [0, 255, 255],  # Caminhão
    13: [255, 0, 0],  # Carro
    128: [128, 128, 128],  # Pista
    129: [0, 165, 255]  # Cones
}


# ---------- Pré-processamento ----------
class FramePreprocessor:
    def __init__(self, width, height):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, frame):
        """Converte BGR → RGB, redimensiona e normaliza."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame_rgb)
            return frame_rgb, frame_tensor
        except Exception as e:
            print(f"Erro no pré-processamento do frame: {e}")
            return None, None


# ---------- Segmentação semântica ----------
class SemanticSegmentation:
    def __init__(self, class_colors):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        #self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(self.device).eval()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(self.device).eval()
        self.class_colors = class_colors.copy()

    def segment_frame(self, frame_tensor):
        with torch.no_grad():
            output = self.model(frame_tensor.unsqueeze(0).to(self.device))['out']
            return torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    def create_segmentation_image(self, seg_map, output_size, original_frame):
        seg_image = np.zeros((*seg_map.shape, 3), dtype=np.uint8)

        for cid in np.unique(seg_map):
            seg_image[seg_map == cid] = self.class_colors.get(cid, [255, 255, 255])

        hsv = cv2.cvtColor(cv2.resize(original_frame, (seg_map.shape[1], seg_map.shape[0])), cv2.COLOR_BGR2HSV)
        pista_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        cone_mask = cv2.inRange(hsv, (10, 100, 100), (30, 255, 255))

        seg_image[(pista_mask > 0) & (seg_map == 0)] = self.class_colors[128]
        seg_image[(cone_mask > 0) & (seg_map == 0)] = self.class_colors[129]

        return cv2.resize(seg_image, output_size, interpolation=cv2.INTER_NEAREST)


# ---------- Rastreador ----------
class SimpleTracker:
    def __init__(self, line_start, line_end, max_distance=50):
        self.line_start = line_start
        self.line_end = line_end
        self.max_distance = max_distance
        self.next_id = 0
        self.objects = {}
        self.prev_objects = {}
        self.counted_ids = set()

    def update(self, detections):
        updated = {}
        used = set()

        for oid, prev_c in self.objects.items():
            if not detections:
                continue
            dists = [np.linalg.norm(np.array(prev_c) - np.array(c)) for c in detections]
            idx = np.argmin(dists)
            if dists[idx] < self.max_distance and idx not in used:
                updated[oid] = detections[idx]
                used.add(idx)
            else:
                updated[oid] = prev_c

        for i, c in enumerate(detections):
            if i not in used:
                updated[self.next_id] = c
                self.next_id += 1

        self.prev_objects = self.objects
        self.objects = updated
        return updated

    def count_crossings(self):
        new_cnt = 0
        for oid, curr in self.objects.items():
            prev = self.prev_objects.get(oid)
            if prev and oid not in self.counted_ids:
                if self._intersect(prev, curr, self.line_start, self.line_end):
                    self.counted_ids.add(oid)
                    new_cnt += 1
        return new_cnt

    @staticmethod
    def _intersect(p1, p2, s1, s2):
        """Verifica se duas linhas se cruzam."""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, s1, s2) != ccw(p2, s1, s2) and ccw(p1, p2, s1) != ccw(p1, p2, s2)


# ---------- Função principal ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Erro ao abrir vídeo.")
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = int(orig_w * RESIZE_FACTOR), int(orig_h * RESIZE_FACTOR)

    preprocessor = FramePreprocessor(width, height)
    segmenter = SemanticSegmentation(CLASS_COLORS)

    tracker = SimpleTracker(line_start=(0, height), line_end=(width, 0), max_distance=60)
    total = 0
    frame_id = 0
    t0 = time.time()

    cv2.namedWindow("Original + Segmentado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original + Segmentado", 1280, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % SKIP_FRAMES != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, (width, height))
        _, frame_tensor = preprocessor.preprocess(frame)
        if frame_tensor is None:
            frame_id += 1
            continue

        seg_map = segmenter.segment_frame(frame_tensor)
        seg_vis = segmenter.create_segmentation_image(seg_map, (width, height), frame)

        centroids = []
        for cid in VEHICLE_CLASSES:
            mask = (seg_map == cid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 200:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                centroids.append((cx, cy))
                cv2.rectangle(seg_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        tracker.update(centroids)
        total += tracker.count_crossings()

        cv2.line(seg_vis, tracker.line_start, tracker.line_end, (0, 0, 255), 2)
        # cv2.putText(seg_vis, f'Veiculos (frame): {len(centroids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(seg_vis, f'Total de Veiculos: {total}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Original + Segmentado", np.hstack((frame, seg_vis)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f'--- Fim ---  Total: {total}  |  Tempo: {time.time() - t0:.1f}s')


if __name__ == "__main__":
    main()
