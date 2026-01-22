"""
xyMath - API de Correção Automática de Gabaritos
Usando OpenCV para processamento de imagem (custo zero)

Deploy: Railway (gratuito)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# Configurações
BUBBLE_THRESHOLD = 0.4
MIN_BUBBLE_AREA = 100
MAX_BUBBLE_AREA = 2000


def decode_base64_image(base64_string):
    """Converte imagem base64 para OpenCV"""
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def find_corner_markers(image):
    """
    Detecta os 4 marcadores de canto (quadrados pretos)
    """
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
        if area < 500 or area > 10000:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            if 0.8 <= aspect_ratio <= 1.2:
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
    
    return result


def read_qr_code(image):
    """
    Lê o QR Code usando OpenCV QRCodeDetector
    Formato esperado: simulado_id|aluno_id|turma_id|total_questoes
    """
    detector = cv2.QRCodeDetector()
    
    # Tentar detectar QR code
    data, vertices, _ = detector.detectAndDecode(image)
    
    if data:
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
    
    # Tentar com imagem em escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data, vertices, _ = detector.detectAndDecode(gray)
    
    if data:
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


def detect_bubbles(image, num_questions, options_per_question=5):
    """
    Detecta as bolhas preenchidas na folha de respostas
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_BUBBLE_AREA < area < MAX_BUBBLE_AREA:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_val = cv2.mean(gray, mask=mask)[0]
                        
                        bubbles.append({
                            'x': cx,
                            'y': cy,
                            'area': area,
                            'intensity': mean_val,
                            'contour': contour
                        })
    
    if not bubbles:
        return [None] * num_questions
    
    bubbles.sort(key=lambda b: b['y'])
    
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    bubble_area_start_y = int(img_height * 0.15)
    bubble_area_height = img_height - bubble_area_start_y
    row_height = bubble_area_height / num_questions
    
    questions = [[] for _ in range(num_questions)]
    
    for bubble in bubbles:
        if bubble['y'] < bubble_area_start_y:
            continue
        
        question_idx = int((bubble['y'] - bubble_area_start_y) / row_height)
        if 0 <= question_idx < num_questions:
            questions[question_idx].append(bubble)
    
    answers = []
    options = ['A', 'B', 'C', 'D', 'E'][:options_per_question]
    
    for q_idx, q_bubbles in enumerate(questions):
        if not q_bubbles:
            answers.append(None)
            continue
        
        q_bubbles.sort(key=lambda b: b['x'])
        
        darkest = min(q_bubbles, key=lambda b: b['intensity'])
        avg_intensity = sum(b['intensity'] for b in q_bubbles) / len(q_bubbles)
        
        if darkest['intensity'] < avg_intensity * 0.7:
            option_width = img_width / options_per_question
            option_idx = int(darkest['x'] / option_width)
            
            if 0 <= option_idx < len(options):
                answers.append(options[option_idx])
            else:
                answers.append(None)
        else:
            answers.append(None)
    
    return answers


def process_answer_sheet(image):
    """Processa uma folha de respostas completa"""
    try:
        # 1. Detectar marcadores de canto
        corners = find_corner_markers(image)
        
        if corners is None:
            # Tentar sem correção de perspectiva
            corrected = image
        else:
            # 2. Corrigir perspectiva
            corrected = perspective_transform(image, corners)
        
        # 3. Ler QR Code
        qr_data = read_qr_code(corrected)
        
        if qr_data is None:
            qr_data = read_qr_code(image)
        
        if qr_data is None:
            return {
                'success': False,
                'error': 'Não foi possível ler o QR Code. Verifique se ele está visível e não danificado.'
            }
        
        # 4. Detectar bolhas preenchidas
        num_questions = qr_data.get('total_questoes', 20)
        answers = detect_bubbles(corrected, num_questions)
        
        return {
            'success': True,
            'qr_data': qr_data,
            'answers': answers,
            'total_detected': len([a for a in answers if a is not None]),
            'total_questions': num_questions
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Erro ao processar imagem: {str(e)}'
        }


# ==================== ROTAS DA API ====================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'xyMath OpenCV API',
        'version': '1.0.0'
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
        return jsonify({
            'success': False,
            'error': f'Erro interno: {str(e)}'
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
        qr_data = read_qr_code(image)
        
        if qr_data:
            return jsonify({
                'success': True,
                'qr_data': qr_data
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
