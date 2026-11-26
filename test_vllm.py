import urllib.request
import json
import time

def test_vllm():
    url = "http://localhost:8888/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are an expert AI assistant running on AWS EKS with vLLM."},
            {"role": "user", "content": "what is capital of Spain  "}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print("[*] Sending request to vLLM on EKS...")
    start_time = time.time()
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            latency = time.time() - start_time
            reply = res_data["choices"][0]["message"]["content"]
            print(f"[+] Response received in {latency:.2f}s:")
            print("-" * 50)
            print(reply)
            print("-" * 50)
    except Exception as e:
        print(f"[-] Request failed: {e}")

if __name__ == "__main__":
    test_vllm()
