"""
xyMath - API de Correção Automática de Gabaritos
Usando OpenCV + pyzbar para processamento de imagem

Deploy: Railway
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os

# Tentar importar pyzbar (melhor para QR Code)
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configurações
BUBBLE_THRESHOLD = 0.5
MIN_BUBBLE_AREA = 80
MAX_BUBBLE_AREA = 3000
FILLED_THRESHOLD = 0.35  # Limiar para considerar bolha preenchida


def decode_base64_image(base64_string):
    """Converte imagem base64 para OpenCV"""
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def read_qr_code(image):
    """
    Lê o QR Code usando pyzbar (preferido) ou OpenCV
    Suporta formato JSON: {"s":"id","a":"id","t":"id","q":15}
    """
    qr_data = None
    qr_location = None
    
    # Converter para escala de cinza
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Método 1: pyzbar (mais confiável)
    if PYZBAR_AVAILABLE:
        # Tentar com imagem original
        decoded = pyzbar.decode(image)
        if not decoded:
            # Tentar com escala de cinza
            decoded = pyzbar.decode(gray)
        if not decoded:
            # Tentar com threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            decoded = pyzbar.decode(thresh)
        if not decoded:
            # Tentar com adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            decoded = pyzbar.decode(thresh)
        
        if decoded:
            data = decoded[0].data.decode('utf-8')
            rect = decoded[0].rect
            qr_location = {
                'x': rect.left,
                'y': rect.top,
                'width': rect.width,
                'height': rect.height
            }
            qr_data = parse_qr_data(data)
    
    # Método 2: OpenCV QRCodeDetector (fallback)
    if qr_data is None:
        detector = cv2.QRCodeDetector()
        
        # Tentar várias versões da imagem
        for img in [image, gray]:
            data, vertices, _ = detector.detectAndDecode(img)
            if data:
                qr_data = parse_qr_data(data)
                if vertices is not None and len(vertices) > 0:
                    pts = vertices[0]
                    x_coords = [p[0] for p in pts]
                    y_coords = [p[1] for p in pts]
                    qr_location = {
                        'x': int(min(x_coords)),
                        'y': int(min(y_coords)),
                        'width': int(max(x_coords) - min(x_coords)),
                        'height': int(max(y_coords) - min(y_coords))
                    }
                break
    
    return qr_data, qr_location


def parse_qr_data(data):
    """
    Parse QR Code data - suporta JSON e formato pipe
    JSON: {"s":"simulado_id","a":"aluno_id","t":"turma_id","q":15}
    Pipe: simulado_id|aluno_id|turma_id|total_questoes
    """
    if not data:
        return None
    
    # Tentar JSON primeiro
    try:
        json_data = json.loads(data)
        return {
            'simulado_id': json_data.get('s') or json_data.get('simulado_id'),
            'aluno_id': json_data.get('a') or json_data.get('aluno_id'),
            'turma_id': json_data.get('t') or json_data.get('turma_id'),
            'total_questoes': int(json_data.get('q') or json_data.get('total_questoes') or 20)
        }
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Tentar formato pipe
    try:
        parts = data.split('|')
        if len(parts) >= 4:
            return {
                'simulado_id': parts[0],
                'aluno_id': parts[1],
                'turma_id': parts[2],
                'total_questoes': int(parts[3])
            }
    except:
        pass
    
    return None


def find_corner_markers(image):
    """Detecta os 4 marcadores de canto (quadrados pretos)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    markers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300 or area > 15000:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            if 0.7 <= aspect_ratio <= 1.3:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    markers.append((cx, cy, area))
    
    markers.sort(key=lambda x: x[2], reverse=True)
    markers = markers[:4]
    
    if len(markers) < 4:
        return None
    
    points = [(m[0], m[1]) for m in markers]
    points = sorted(points, key=lambda p: p[1])
    top_points = sorted(points[:2], key=lambda p: p[0])
    bottom_points = sorted(points[2:], key=lambda p: p[0])
    
    return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]


def perspective_transform(image, corners):
    """Corrige a perspectiva da imagem"""
    pts1 = np.float32(corners)
    
    width = max(
        np.linalg.norm(np.array(corners[0]) - np.array(corners[1])),
        np.linalg.norm(np.array(corners[3]) - np.array(corners[2]))
    )
    height = max(
        np.linalg.norm(np.array(corners[0]) - np.array(corners[3])),
        np.linalg.norm(np.array(corners[1]) - np.array(corners[2]))
    )
    
    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (int(width), int(height)))
    
    return result, matrix
    def detect_bubbles(image, num_questions, options_per_question=5):
    """
    Detecta as bolhas preenchidas na folha de respostas
    Retorna: answers (lista), bubble_locations (coordenadas para overlay)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Threshold adaptativo para melhor detecção
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_height, img_width = image.shape[:2]
    
    # Filtrar bolhas por tamanho e circularidade
    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_BUBBLE_AREA < area < MAX_BUBBLE_AREA:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:  # Círculos têm circularity ~1.0
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calcular preenchimento (intensidade média dentro da bolha)
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_val = cv2.mean(gray, mask=mask)[0]
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        bubbles.append({
                            'x': cx,
                            'y': cy,
                            'area': area,
                            'intensity': mean_val,
                            'rect': {'x': x, 'y': y, 'w': w, 'h': h},
                            'contour': contour
                        })
    
    if not bubbles:
        return [None] * num_questions, []
    
    # Área das bolhas (excluir cabeçalho ~20% superior)
    bubble_area_start_y = int(img_height * 0.20)
    bubble_area_end_y = int(img_height * 0.95)
    bubble_area_height = bubble_area_end_y - bubble_area_start_y
    
    # Calcular questões por coluna (layout 2 colunas)
    questions_per_column = (num_questions + 1) // 2
    row_height = bubble_area_height / questions_per_column
    column_width = img_width / 2
    
    # Organizar bolhas por questão
    questions = [[] for _ in range(num_questions)]
    
    for bubble in bubbles:
        # Ignorar bolhas fora da área de respostas
        if bubble['y'] < bubble_area_start_y or bubble['y'] > bubble_area_end_y:
            continue
        
        # Determinar coluna (esquerda ou direita)
        col = 0 if bubble['x'] < column_width else 1
        
        # Determinar linha dentro da coluna
        relative_y = bubble['y'] - bubble_area_start_y
        row = int(relative_y / row_height)
        
        # Calcular índice da questão
        if col == 0:
            question_idx = row
        else:
            question_idx = questions_per_column + row
        
        if 0 <= question_idx < num_questions:
            questions[question_idx].append(bubble)
    
    # Processar cada questão
    answers = []
    bubble_locations = []
    options = ['A', 'B', 'C', 'D', 'E'][:options_per_question]
    
    for q_idx, q_bubbles in enumerate(questions):
        if not q_bubbles:
            answers.append(None)
            bubble_locations.append({
                'question': q_idx + 1,
                'answer': None,
                'status': 'not_found',
                'bubbles': []
            })
            continue
        
        # Ordenar bolhas da esquerda para direita
        q_bubbles.sort(key=lambda b: b['x'])
        
        # Calcular intensidade média de todas as bolhas da questão
        avg_intensity = sum(b['intensity'] for b in q_bubbles) / len(q_bubbles)
        
        # Encontrar bolhas preenchidas (mais escuras que a média)
        filled_bubbles = []
        for i, bubble in enumerate(q_bubbles):
            # Bolha é considerada preenchida se for significativamente mais escura
            if bubble['intensity'] < avg_intensity * (1 - FILLED_THRESHOLD):
                filled_bubbles.append((i, bubble))
        
        # Preparar dados de localização
        bubbles_info = []
        for i, bubble in enumerate(q_bubbles):
            option_letter = options[i] if i < len(options) else '?'
            is_filled = any(fb[0] == i for fb in filled_bubbles)
            bubbles_info.append({
                'option': option_letter,
                'x': bubble['x'],
                'y': bubble['y'],
                'rect': bubble['rect'],
                'filled': is_filled,
                'intensity': bubble['intensity']
            })
        
        # Determinar resposta
        if len(filled_bubbles) == 0:
            # Nenhuma bolha preenchida = em branco
            answers.append(None)
            status = 'blank'
            answer = None
        elif len(filled_bubbles) == 1:
            # Exatamente uma bolha preenchida = resposta válida
            idx = filled_bubbles[0][0]
            if idx < len(options):
                answer = options[idx]
                answers.append(answer)
                status = 'valid'
            else:
                answers.append(None)
                answer = None
                status = 'invalid'
        else:
            # Múltiplas bolhas preenchidas = ANULADA
            answers.append('X')  # X indica múltipla marcação (anulada)
            answer = 'X'
            status = 'multiple'
        
        bubble_locations.append({
            'question': q_idx + 1,
            'answer': answer,
            'status': status,
            'bubbles': bubbles_info
        })
    
    return answers, bubble_locations


def process_answer_sheet(image):
    """Processa uma folha de respostas completa"""
    try:
        original_image = image.copy()
        img_height, img_width = image.shape[:2]
        
        # 1. Tentar ler QR Code primeiro (antes de qualquer transformação)
        qr_data, qr_location = read_qr_code(image)
        
        # 2. Detectar marcadores de canto
        corners = find_corner_markers(image)
        transform_matrix = None
        
        if corners is not None:
            # 3. Corrigir perspectiva
            corrected, transform_matrix = perspective_transform(image, corners)
            
            # Tentar ler QR novamente na imagem corrigida
            if qr_data is None:
                qr_data, qr_location = read_qr_code(corrected)
        else:
            corrected = image
        
        # Se ainda não leu o QR Code, retornar erro
        if qr_data is None:
            return {
                'success': False,
                'error': 'QR Code não encontrado. Verifique se está visível e bem iluminado.',
                'corners_found': corners is not None,
                'qr_location': qr_location
            }
        
        # 4. Detectar bolhas preenchidas
        num_questions = qr_data.get('total_questoes', 20)
        answers, bubble_locations = detect_bubbles(corrected, num_questions)
        
        # Contar estatísticas
        total_detected = len([a for a in answers if a is not None])
        total_blank = len([a for a in answers if a is None])
        total_multiple = len([a for a in answers if a == 'X'])
        total_valid = len([a for a in answers if a is not None and a != 'X'])
        
        return {
            'success': True,
            'qr_data': qr_data,
            'qr_location': qr_location,
            'answers': answers,
            'bubble_locations': bubble_locations,
            'total_questions': num_questions,
            'total_detected': total_detected,
            'total_valid': total_valid,
            'total_blank': total_blank,
            'total_multiple': total_multiple,
            'corners_found': corners is not None,
            'image_size': {'width': img_width, 'height': img_height}
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f'Erro ao processar imagem: {str(e)}',
            'traceback': traceback.format_exc()
        }


# ==================== ROTAS DA API ====================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'xyMath OpenCV API',
        'version': '2.0.0',
        'pyzbar_available': PYZBAR_AVAILABLE,
        'features': [
            'QR Code JSON format',
            'Multiple marking detection',
            'Bubble location overlay',
            'Perspective correction'
        ]
    })


@app.route('/api/process', methods=['POST'])
def process_image():
    """Endpoint principal para processar imagem de gabarito"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Imagem não fornecida'
            }), 400
        
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Não foi possível decodificar a imagem'
            }), 400
        
        result = process_answer_sheet(image)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': f'Erro interno: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/detect-qr', methods=['POST'])
def detect_qr_only():
    """Endpoint para detectar apenas o QR Code"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Imagem não fornecida'
            }), 400
        
        image = decode_base64_image(data['image'])
        qr_data, qr_location = read_qr_code(image)
        
        if qr_data:
            return jsonify({
                'success': True,
                'qr_data': qr_data,
                'qr_location': qr_location
            })
        else:
            return jsonify({
                'success': False,
                'error': 'QR Code não encontrado'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/scan-frame', methods=['POST'])
def scan_frame():
    """
    Endpoint para leitura contínua (streaming)
    Retorna rapidamente se detectou QR Code ou não
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'detected': False})
        
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({'detected': False})
        
        # Apenas detectar QR Code (rápido)
        qr_data, qr_location = read_qr_code(image)
        
        if qr_data:
            # QR Code encontrado - processar completo
            result = process_answer_sheet(image)
            result['detected'] = True
            return jsonify(result)
        else:
            return jsonify({
                'detected': False,
                'message': 'Posicione o QR Code na câmera'
            })
            
    except Exception as e:
        return jsonify({
            'detected': False,
            'error': str(e)
        })


@app.route('/api/batch', methods=['POST'])
def batch_process():
    """Endpoint para processar múltiplas imagens"""
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({
                'success': False,
                'error': 'Imagens não fornecidas'
            }), 400
        
        results = []
        for i, img_base64 in enumerate(data['images']):
            try:
                image = decode_base64_image(img_base64)
                result = process_answer_sheet(image)
                result['index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results),
            'processed': len([r for r in results if r.get('success')])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
