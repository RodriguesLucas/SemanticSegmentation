"""
Pipeline de segmentação + rastreamento + contagem em vídeo (praças de pedágio)

Fluxo:
1) Lê frames do vídeo e reduz resolução para ganhar FPS.
2) Pré-processa (BGR→RGB, resize, normalização ImageNet) e roda segmentação (DeepLabV3-ResNet101).
3) Constrói máscaras para as classes de interesse e extrai contornos.
4) Para cada contorno válido, calcula bbox e centróide → gera detecções [(cx,cy),(x,y,w,h)].
5) Rastreia objetos por associação do centróide (distância euclidiana mínima) entre frames.
6) Conta quando a BORDA definida (front/rear) cruza uma linha horizontal de contagem.
7) Renderiza overlay (bboxes, linha, contagem total) e exibe lado a lado.

Observações-chave:
- A contagem depende da **interseção da borda do bbox com a linha**. Com carros descendo
  (de cima→baixo), a “front” é o **TOPO** do bbox e a “rear” é a **BASE** (y+h).
- Para evitar dupla contagem do mesmo veículo na mesma linha, usa-se um registro de IDs já contados.
- Heurística HSV marca “pista” (classe 128) em áreas claras rotuladas originalmente como fundo (classe 0).
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torchvision import models, transforms

# ---------- Configurações ----------
# Caminho do vídeo (mantém 3 opções; ative apenas uma)
# VIDEO_PATH = "videos/Vídeo 1 - 14_06_2024 - 16h até 16h20.mp4"
VIDEO_PATH = "videos/Vídeo 2 - 14_06_2024 - 16h20 até 16h40.mp4"
# VIDEO_PATH = "videos/Vídeo 3 - 14_06_2024 - 16h40 até 17h.mp4"

RESIZE_FACTOR = 0.5  # Redução do frame para ganhar FPS (↓ custo na segmentação e desenho)
SKIP_FRAMES = 1  # Processa 1 a cada N frames (↑ N → ↑ FPS, porém ↓ resolução temporal)
VEHICLE_CLASSES = [7]  # Lista de IDs de classe de interesse (ex.: 7 = carro, conforme seu rótulo)
COUNT_EDGE = "rear"  # Qual borda vai disparar a contagem: "front" (frente) ou "rear" (traseira)
# Com veículos descendo: front=TOPO (y), rear=BASE (y+h)

# Paleta de cores (RGB) para visualização das classes segmentadas.
# Observação: o ID 128 (pista) é "artificial", aplicado por heurística HSV sobre pixels do rótulo 0 (fundo).
CLASS_COLORS = {
    0: [0, 0, 0],  # Fundo (preto)
    7: [255, 255, 0],  # Carro (amarelo)
    128: [128, 128, 128],  # Pista (cinza)
}

# ---------- Linha de contagem (em % do frame redimensionado) ----------
# Define uma linha horizontal (y constante) em coordenadas normalizadas [0..1].
# A linha é convertida para pixels após conhecer (width,height) do frame reduzido.
# Ajuste y (0.48) para “adiantar/atrasar” o gatilho conforme seu enquadramento.
LINE1_PCT = ((0.0, 0.48), (0.98, 0.48))


# ---------- Pré-processamento ----------
class FramePreprocessor:
    """
    Converte frame BGR (OpenCV) → RGB, aplica resize e normalização de acordo com ImageNet.
    Isso garante que o tensor esteja no formato/tamanho esperado pelo modelo do Torchvision.
    """

    def __init__(self, width, height) -> None:
        # Compose de transformações: PIL → Resize → Tensor → Normalize
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),  # garante forma (H,W) usada também na visualização
            transforms.ToTensor(),  # escala para [0,1] e troca para (C,H,W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # estatísticas do ImageNet
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        """
        Recebe um frame BGR (H x W x 3) do OpenCV e retorna:
          - frame_rgb: imagem RGB (np.ndarray) já redimensionada
          - frame_tensor: tensor normalizado (3 x H' x W') pronto para a rede
        Em caso de erro, retorna (None, None).
        """
        try:
            # OpenCV lê em BGR; convertemos para RGB para casar com o modelo
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame_rgb)
            return frame_rgb, frame_tensor
        except Exception as e:
            # Falhas pontuais não devem parar o loop; loga e segue
            print(f"Erro no pré-processamento do frame: {e}")
            return None, None


# ---------- Segmentação ----------
class SemanticSegmentation:
    """
    Encapsula o modelo DeepLabV3-ResNet101 (Torchvision) e a lógica de:
      - Seleção de dispositivo (CUDA/CPU)
      - Forward pass (gera mapa de classes por pixel)
      - Criação de uma imagem de visualização colorida (paleta + heurística HSV para pista)
    """

    def __init__(self, class_colors: Dict[int, Sequence[int]]) -> None:
        # Seleciona GPU se disponível; se não, CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        # Carrega pesos (API nova "weights" ou fallback "pretrained=True")
        self.model = self._load_model().to(self.device).eval()
        # Paleta defensiva (cópia) para evitar mutações externas
        self.class_colors = dict(class_colors)

    @staticmethod
    def _load_model():
        """
        Tenta a API nova do Torchvision (weights=DEFAULT).
        Se não existir (versões antigas), cai no parâmetro legacy pretrained=True.
        """
        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
            model = models.segmentation.deeplabv3_resnet101(weights=weights)
        except Exception:
            model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        return model

    def segment_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        Faz a inferência no frame e retorna seg_map (H x W) com IDs de classe (argmax dos logits).
        """
        with torch.no_grad():
            # A rede espera batch → unsqueeze(0); pega a saída 'out'
            output = self.model(frame_tensor.unsqueeze(0).to(self.device))['out']
            seg_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
            return seg_map

    def create_segmentation_image(
            self,
            seg_map: np.ndarray,
            output_size: Tuple[int, int],
            original_frame_bgr_resized: np.ndarray
    ) -> np.ndarray:
        """
        Constrói uma imagem RGB colorida a partir de seg_map:
          1) Preenche cores por ID usando CLASS_COLORS (desconhecidas → branco).
          2) Converte o frame original para HSV e aplica uma máscara para tons claros (pista).
          3) Em regiões de fundo (ID 0) que forem claras, força cor de pista (ID 128).
          4) Faz resize final para (output_size) com NEAREST (mantém blocos nítidos).
        """
        # Imagem colorida da segmentação (inicialmente preta)
        seg_image = np.zeros((*seg_map.shape, 3), dtype=np.uint8)

        # Pinta cada ID presente com a cor definida (ou branco, se não tiver na paleta)
        for cid in np.unique(seg_map):
            seg_image[seg_map == cid] = self.class_colors.get(cid, [255, 255, 255])

        # Reforço HSV para "pista" (tons claros, baixa saturação) sobre regiões rotuladas como fundo (0)
        hsv = cv2.cvtColor(
            cv2.resize(original_frame_bgr_resized, (seg_map.shape[1], seg_map.shape[0])),
            cv2.COLOR_BGR2HSV
        )
        # Faixa HSV (H 0..180, S 0..50, V 200..255) → superfícies claras/cinzas (ajuste se necessário)
        pista_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        # Apenas onde a rede disse "fundo" (0) e a heurística sugere pista → pinta como 128
        seg_image[(pista_mask > 0) & (seg_map == 0)] = self.class_colors[128]

        # Resize de volta ao tamanho de saída para casar com o frame exibido
        return cv2.resize(seg_image, output_size, interpolation=cv2.INTER_NEAREST)


# ---------- Rastreador (conta pela BORDA escolhida) ----------
class MultiLineTracker:
    """
    Rastreia objetos por associação de centróides (distância mínima) e realiza contagem
    quando a **borda** definida cruza uma **linha horizontal**.

    Direções e bordas:
      - Carros descendo (dy > 0):
          front = TOPO do bbox (y)
          rear  = BASE do bbox (y + h)
      - Carros subindo (dy < 0):
          front = BASE do bbox (y + h)
          rear  = TOPO do bbox (y)

    COUNT_EDGE: "front" ou "rear" decide qual borda será comparada com a linha.

    Estruturas principais:
      - self.objects = { id: {'c':(cx,cy), 'bbox':(x,y,w,h)} }   # estado atual
      - self.prev_objects = idem                                  # estado anterior
      - self.counted_ids_per_line = [ set(), ... ]                # evita duplicidade por linha

    Observações:
      - Associação gulosa por menor distância pode trocar IDs em oclusões/saltos.
      - 'max_distance' define tolerância para match; subir demais pode gerar mismatches.
    """

    def __init__(self, segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                 max_distance: float = 60.0,
                 count_edge: str = "rear") -> None:
        self.segments = segments  # lista de segmentos de linha em pixels
        self.max_distance = max_distance  # raio máx. para associar centróides
        self.count_edge = count_edge  # "front" (frente) ou "rear" (traseira)
        self.next_id = 0  # gerador de IDs
        self.objects = {}  # estado atual por ID
        self.prev_objects = {}  # estado anterior por ID
        self.counted_ids_per_line = [set() for _ in segments]  # marca IDs já contados

    def update(self, detections: List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]]) -> Dict[int, Dict]:
        """
        Atualiza rastros com base nas detecções atuais.
        Estratégia:
          - Para cada ID antigo, procura a detecção mais próxima (centróide) < max_distance.
          - Detecções não associadas viram novos IDs.
          - Se não houver detecções, mantém estado anterior (pode segurar um frame).
        """
        updated = {}
        used = set()

        for oid, prev_state in self.objects.items():
            if not detections:
                continue
            prev_c = prev_state['c']
            # Distâncias euclidianas do centróide anterior para todos os atuais
            dists = [np.linalg.norm(np.array(prev_c) - np.array(c)) for (c, _) in detections]
            idx = int(np.argmin(dists))
            # Associa se estiver dentro do raio e aquela detecção ainda não foi usada
            if dists[idx] < self.max_distance and idx not in used:
                c, bbox = detections[idx]
                updated[oid] = {'c': c, 'bbox': bbox}
                used.add(idx)
            else:
                # Sem associação → mantém posição anterior (reduz “sumiços” de 1 frame)
                updated[oid] = prev_state

        # Cria novos IDs para detecções remanescentes (não associadas)
        for i, (c, bbox) in enumerate(detections):
            if i not in used:
                updated[self.next_id] = {'c': c, 'bbox': bbox}
                self.next_id += 1

        # Avança estados
        self.prev_objects = self.objects
        self.objects = updated
        return updated

    def count_crossings(self) -> int:
        """
        Verifica cruzamento da borda escolhida contra cada linha HORIZONTAL.
        Critério de cruzamento:
          - crossed_down: edge_prev < line_y <= edge_curr (indo para baixo)
          - crossed_up:   edge_prev > line_y >= edge_curr (indo para cima)
        O uso de desigualdade estrita/inclusiva evita perder o instante de toque na linha.
        """
        new_total = 0
        for li, (s0, s1) in enumerate(self.segments):
            # Este modo assume linhas horizontais (y constante)
            if s0[1] != s1[1]:
                continue
            line_y = s0[1]

            for oid, curr in self.objects.items():
                prev = self.prev_objects.get(oid)
                # Precisa ter histórico e não ter sido contado nessa linha
                if not prev or oid in self.counted_ids_per_line[li]:
                    continue

                # Direção do movimento no eixo y (positiva = descendo)
                (_, cy_prev) = prev['c']
                (_, cy_curr) = curr['c']
                dy = cy_curr - cy_prev  # >0 descendo, <0 subindo

                # Borda superior/inferior do bbox (no frame anterior e atual)
                (x_prev, y_prev, w_prev, h_prev) = prev['bbox']
                (x_curr, y_curr, w_curr, h_curr) = curr['bbox']

                if dy > 0:  # descendo
                    # Frente = topo; Traseira = base
                    front_prev, front_curr = y_prev, y_curr
                    rear_prev, rear_curr = y_prev + h_prev, y_curr + h_curr
                elif dy < 0:  # subindo
                    # Frente = base; Traseira = topo
                    front_prev, front_curr = y_prev + h_prev, y_curr + h_curr
                    rear_prev, rear_curr = y_prev, y_curr
                else:
                    # Sem deslocamento vertical suficiente → ignora
                    continue

                # Seleciona a borda que será comparada com a linha
                edge_prev = front_prev if self.count_edge == "front" else rear_prev
                edge_curr = front_curr if self.count_edge == "front" else rear_curr

                # Cruzamento: mudou de lado em relação à linha (considera direção)
                crossed_down = (edge_prev < line_y <= edge_curr)  # para baixo
                crossed_up = (edge_prev > line_y >= edge_curr)  # para cima

                if crossed_down or crossed_up:
                    # Marca para não contar de novo este ID nesta linha
                    self.counted_ids_per_line[li].add(oid)
                    new_total += 1

        return new_total


# ---------- Função principal ----------
def main() -> None:
    """
    Orquestra o pipeline:
      - Abre o vídeo e calcula dimensões reduzidas
      - Constrói: pré-processador, segmentador e rastreador
      - Loop:
          * (opcional) pula frames para ganhar FPS
          * resize, pré-processa, segmenta
          * extrai contornos → centróides + bboxes
          * rastreia e conta cruzamentos
          * desenha overlays e exibe
      - Finaliza liberando recursos
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Erro ao abrir vídeo.")
        sys.exit(1)

    # Dimensões originais → aplica redução para performance
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = int(orig_w * RESIZE_FACTOR), int(orig_h * RESIZE_FACTOR)

    # Helper: converte coordenadas em % (0..1) para pixels na resolução atual
    def pct_to_px(pct_pair: Tuple[Tuple[float, float], Tuple[float, float]]
                  ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        (x1p, y1p), (x2p, y2p) = pct_pair
        return (int(x1p * width), int(y1p * height)), (int(x2p * width), int(y2p * height))

    # Converte a linha de contagem para pixels (na resolução reduzida)
    line1 = pct_to_px(LINE1_PCT)

    # Instancia componentes do pipeline
    preprocessor = FramePreprocessor(width, height)
    segmenter = SemanticSegmentation(CLASS_COLORS)
    tracker = MultiLineTracker(segments=[line1], max_distance=60.0, count_edge=COUNT_EDGE)

    total = 0  # acumulador de contagens
    frame_id = 0  # índice de frames (após SKIP_FRAMES)
    t0 = time.time()

    # Janela de visualização: mostra Original | Segmentado lado a lado
    cv2.namedWindow("Original + Segmentado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original + Segmentado", 1280, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pula frames para ganhar FPS (cuidado: reduz chance de detectar cruzamentos estreitos)
        if frame_id % SKIP_FRAMES != 0:
            frame_id += 1
            continue

        # Redimensiona o frame bruto e pré-processa para o modelo
        frame = cv2.resize(frame, (width, height))
        _, frame_tensor = preprocessor.preprocess(frame)
        if frame_tensor is None:
            frame_id += 1
            continue

        # Segmentação: retorna seg_map (H x W) com IDs
        seg_map = segmenter.segment_frame(frame_tensor)
        # Cria visual de segmentação para overlay (cores + heurística de pista)
        seg_vis = segmenter.create_segmentation_image(seg_map, (width, height), frame)

        # ---------- Detecção: centróide + bbox por classe alvo ----------
        # Estratégia simples: binariza cada classe → contornos → filtra por área mínima
        detections = []
        for cid in VEHICLE_CLASSES:
            mask = (seg_map == cid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                # Remove ruído (blobs muito pequenos não viram veículo)
                if cv2.contourArea(cnt) < 250:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                detections.append(((cx, cy), (x, y, w, h)))
                # Opcional: desenhar bbox no overlay para depuração
                cv2.rectangle(seg_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ---------- Rastreamento + Contagem ----------
        tracker.update(detections)  # associa detecções a IDs previamente rastreados
        total += tracker.count_crossings()  # incrementa quando a borda cruza a linha

        # Desenha linha(s) de contagem
        for (p0, p1) in tracker.segments:
            cv2.line(seg_vis, p0, p1, (0, 0, 255), 2)

        # HUD com total acumulado
        cv2.putText(seg_vis, f'Total de Veiculos: {total}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Mostra original e segmentado lado a lado
        cv2.imshow("Original + Segmentado", np.hstack((frame, seg_vis)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    # Liberação de recursos
    cap.release()
    cv2.destroyAllWindows()
    print(f'--- Fim ---  Total: {total}  |  Tempo: {time.time() - t0:.1f}s')


if __name__ == "__main__":
    main()
