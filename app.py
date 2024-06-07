import os
import hashlib
import time
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def upload_file_to_virustotal(file_path):
    url = "https://www.virustotal.com/vtapi/v2/file/scan"
    file_hash = get_file_hash(file_path)
    params = {
        "apikey": API_KEY
    }
    files = {
        "file": (file_hash, open(file_path, "rb"))
    }
    response = requests.post(url, files=files, params=params)
    return response.json()

def get_file_report(scan_id):
    url = f"https://www.virustotal.com/vtapi/v2/file/report"
    params = {
        "apikey": API_KEY,
        "resource": scan_id
    }
    response = requests.get(url, params=params)
    return response.json()

def scan_url_with_virustotal(url_to_scan):
    url = "https://www.virustotal.com/vtapi/v2/url/scan"
    params = {
        "apikey": API_KEY,
        "url": url_to_scan
    }
    response = requests.post(url, data=params)
    return response.json()

def get_url_report(scan_id):
    url = f"https://www.virustotal.com/vtapi/v2/url/report"
    params = {
        "apikey": API_KEY,
        "resource": scan_id
    }
    response = requests.get(url, params=params)
    return response.json()

@app.route('/scan_file', methods=['POST'])
def scan_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)

    response = upload_file_to_virustotal(file_path)
    if response["response_code"] == 1:
        scan_id = response["scan_id"]
        while True:
            report = get_file_report(scan_id)
            if report["response_code"] == 1:
                positives = report["positives"]
                total = report["total"]
                malware_ratio = positives / total

                if malware_ratio > 0.5:
                    result = "File is likely malware."
                else:
                    result = "File is likely not malware."

                return jsonify({
                    'result': result,
                    'detection_ratio': f"{malware_ratio:.2f} ({positives}/{total})"
                })
            elif report["response_code"] == -2:
                time.sleep(60)
            else:
                return jsonify({'error': report["verbose_msg"]}), 400

    return jsonify({'error': response["verbose_msg"]}), 400

@app.route('/scan_url', methods=['POST'])
def scan_url():
    data = request.get_json()
    url_to_scan = data.get('url')
    if not url_to_scan:
        return jsonify({'error': 'No URL provided'}), 400

    response = scan_url_with_virustotal(url_to_scan)
    if response["response_code"] == 1:
        scan_id = response["scan_id"]
        while True:
            report = get_url_report(scan_id)
            if report["response_code"] == 1:
                positives = report["positives"]
                total = report["total"]
                if positives > 0:
                    result = "URL is likely malicious."
                else:
                    result = "URL is likely safe."

                return jsonify({
                    'result': result,
                    'detection_ratio': f"{positives}/{total}"
                })
            elif report["response_code"] == -2:
                time.sleep(60)
            else:
                return jsonify({'error': report["verbose_msg"]}), 400

    return jsonify({'error': response["verbose_msg"]}), 400

if __name__ == '__main__':
    app.run(debug=True)

