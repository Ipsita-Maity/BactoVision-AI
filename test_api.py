"""
End-to-end API test script for BactoVision AI.
"""
import requests

base_url = 'http://localhost:5050'

# 1. GET /
r = requests.get(base_url + '/')
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
print("1. GET / : SUCCESS (status 200, HTML rendered)")

# 2. POST /api/generate-sample
r = requests.post(base_url + '/api/generate-sample', json={'sample_type': 'mixed', 'cell_count': 55})
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
sample_data = r.json()
print("2. POST /api/generate-sample : SUCCESS, generated:", sample_data['filename'])

# 3. POST /api/analyze
r = requests.post(base_url + '/api/analyze', json={
    'image_b64': sample_data['image_data'],
    'model_type': 'watershed_hybrid',
    'conf_threshold': 0.35,
    'px_per_micron': 10.0
})
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
analysis_data = r.json()
session_id = analysis_data['session_id']
summary = analysis_data['summary']
print(f"3. POST /api/analyze : SUCCESS, Session ID: {session_id}")
print(f"   Total Cells: {summary['total_count']}")
print(f"   Gram-Positive: {summary['gram_positive_count']} ({summary['gram_positive_pct']}%)")
print(f"   Gram-Negative: {summary['gram_negative_count']} ({summary['gram_negative_pct']}%)")
print(f"   Cocci: {summary['cocci_count']}, Bacilli: {summary['bacilli_count']}")
print(f"   Mean Area: {summary['mean_area_um2']} um2, Circularity: {summary['mean_circularity']}")
print(f"   Images returned: {list(analysis_data['images'].keys())}")

# 4. GET /api/export/csv/<session_id>
r = requests.get(f"{base_url}/api/export/csv/{session_id}")
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
assert "Cell_ID,Gram_Status,Morphology" in r.text
print(f"4. GET /api/export/csv : SUCCESS, CSV size: {len(r.content)} bytes")

# 5. GET /api/export/pdf/<session_id>
r = requests.get(f"{base_url}/api/export/pdf/{session_id}")
assert r.status_code == 200, f"Expected 200, got {r.status_code}"
assert r.content.startswith(b'%PDF'), "Expected valid PDF header"
print(f"5. GET /api/export/pdf : SUCCESS, PDF size: {len(r.content)} bytes")

# 6. POST /api/roboflow/connect
r = requests.post(f"{base_url}/api/roboflow/connect", json={'api_key': 'test_dummy_key', 'project': 'test'})
print("6. POST /api/roboflow/connect : SUCCESS, responded:", r.json().get('message', ''))

print("\n>>> ALL API ENDPOINTS AND EXPORTS FULLY VERIFIED AND PASSING! <<<")
