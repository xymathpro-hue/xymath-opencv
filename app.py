"""
xyMath - API de Correção Automática de Gabaritos
Usando OpenCV para processamento de imagem

Deploy: Railway
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os

app = Flask(__name__)
CORS(app)

# Configurações
MIN_BUBBLE_AREA = 80
MAX_BUBBLE_AREA = 3000
FILLED_THRESHOLD = 0.35


def decode_base64_image(base64_string):
    """Converte imagem base64 para OpenCV"""
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def read_qr_code(image):
    """Lê QR Code usando OpenCV - suporta JSON"""
    detector = cv2.QRCodeDetector()
    
    # Tentar com imagem original
    data, vertices, _ = detector.detectAndDecode(image)
    
    if not data:
        # Tentar com escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data, vertices, _ = detector.detectAndDecode(gray)
    
    if not data:
        # Tentar com threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        data, vertices, _ = detector.detectAndDecode(thresh)
    
    if not data:
        # Tentar com contraste aumentado
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        data, vertices, _ = detector.detectAndDecode(enhanced)
    
    if data:
        qr_location = None
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
        return parse_qr_data(data), qr_location
    
    return None, None


def parse_qr_data(data):
    """Parse QR Code - suporta JSON e formato pipe"""
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
    except:
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
    """Detecta os 4 marcadores de canto"""
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
    
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (int(width), int(height)))
    
    return result


def detect_bubbles(image, num_questions, options_per_question=5):
    """Detecta bolhas preenchidas"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_height, img_width = image.shape[:2]
    
    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_BUBBLE_AREA < area < MAX_BUBBLE_AREA:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_val = cv2.mean(gray, mask=mask)[0]
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        bubbles.append({
                            'x': cx, 'y': cy, 'area': area,
                            'intensity': mean_val,
                            'rect': {'x': x, 'y': y, 'w': w, 'h': h}
                        })
    
    if not bubbles:
        return [None] * num_questions, []
    
    # Layout 2 colunas
    bubble_area_start_y = int(img_height * 0.20)
    bubble_area_end_y = int(img_height * 0.95)
    bubble_area_height = bubble_area_end_y - bubble_area_start_y
    
    questions_per_column = (num_questions + 1) // 2
    row_height = bubble_area_height / questions_per_column
    column_width = img_width / 2
    
    questions = [[] for _ in range(num_questions)]
    
    for bubble in bubbles:
        if bubble['y'] < bubble_area_start_y or bubble['y'] > bubble_area_end_y:
            continue
        
        col = 0 if bubble['x'] < column_width else 1
        relative_y = bubble['y'] - bubble_area_start_y
        row = int(relative_y / row_height)
        
        question_idx = row if col == 0 else questions_per_column + row
        
        if 0 <= question_idx < num_questions:
            questions[question_idx].append(bubble)
    
    answers = []
    bubble_locations = []
    options = ['A', 'B', 'C', 'D', 'E'][:options_per_question]
    
    for q_idx, q_bubbles in enumerate(questions):
        if not q_bubbles:
            answers.append(None)
            bubble_locations.append({'question': q_idx + 1, 'answer': None, 'status': 'not_found', 'bubbles': []})
            continue
        
        q_bubbles.sort(key=lambda b: b['x'])
        avg_intensity = sum(b['intensity'] for b in q_bubbles) / len(q_bubbles)
        
        filled = [(i, b) for i, b in enumerate(q_bubbles) if b['intensity'] < avg_intensity * (1 - FILLED_THRESHOLD)]
        
        bubbles_info = []
        for i, bubble in enumerate(q_bubbles):
            opt = options[i] if i < len(options) else '?'
            is_filled = any(f[0] == i for f in filled)
            bubbles_info.append({'option': opt, 'x': bubble['x'], 'y': bubble['y'], 'rect': bubble['rect'], 'filled': is_filled})
        
        if len(filled) == 0:
            answers.append(None)
            status = 'blank'
            answer = None
        elif len(filled) == 1:
            idx = filled[0][0]
            answer = options[idx] if idx < len(options) else None
            answers.append(answer)
            status = 'valid'
        else:
            answers.append('X')
            answer = 'X'
            status = 'multiple'
        
        bubble_locations.append({'question': q_idx + 1, 'answer': answer, 'status': status, 'bubbles': bubbles_info})
    
    return answers, bubble_locations


def process_answer_sheet(image):
    """Processa folha de respostas completa"""
    try:
        img_height, img_width = image.shape[:2]
        
        qr_data, qr_location = read_qr_code(image)
        
        corners = find_corner_markers(image)
        
        if corners is not None:
            corrected = perspective_transform(image, corners)
            if qr_data is None:
                qr_data, qr_location = read_qr_code(corrected)
        else:
            corrected = image
        
        if qr_data is None:
            return {
                'success': False,
                'error': 'QR Code não encontrado. Verifique iluminação e posição.',
                'corners_found': corners is not None
            }
        
        num_questions = qr_data.get('total_questoes', 20)
        answers, bubble_locations = detect_bubbles(corrected, num_questions)
        
        return {
            'success': True,
            'qr_data': qr_data,
            'qr_location': qr_location,
            'answers': answers,
            'bubble_locations': bubble_locations,
            'total_questions': num_questions,
            'total_detected': len([a for a in answers if a is not None]),
            'total_valid': len([a for a in answers if a and a != 'X']),
            'total_blank': len([a for a in answers if a is None]),
            'total_multiple': len([a for a in answers if a == 'X']),
            'corners_found': corners is not None,
            'image_size': {'width': img_width, 'height': img_height}
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Erro: {str(e)}'}


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'xyMath OpenCV API',
        'version': '2.0.0',
        'features': ['QR Code JSON', 'Multiple marking detection', 'Bubble overlay']
    })


@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Imagem não fornecida'}), 400
        
        image = decode_base64_image(data['image'])
        if image is None:
            return jsonify({'success': False, 'error': 'Erro ao decodificar imagem'}), 400
        
        return jsonify(process_answer_sheet(image))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect-qr', methods=['POST'])
def detect_qr_only():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Imagem não fornecida'}), 400
        
        image = decode_base64_image(data['image'])
        qr_data, qr_location = read_qr_code(image)
        
        if qr_data:
            return jsonify({'success': True, 'qr_data': qr_data, 'qr_location': qr_location})
        return jsonify({'success': False, 'error': 'QR Code não encontrado'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
