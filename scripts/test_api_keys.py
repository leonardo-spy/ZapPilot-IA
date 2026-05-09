import os
import sys
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Adiciona o diretório raiz ao path para carregar o .env corretamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

def print_separator():
    print("-" * 50)

def test_groq_keys():
    print("\n" + "="*50)
    print("🚀 TESTANDO CHAVES GROQ (LLM)")
    print("="*50)
    
    groq_keys = os.getenv("GROQ_API_KEY", "")
    if not groq_keys:
        print("❌ Nenhuma GROQ_API_KEY encontrada no .env")
        return

    keys = [k.strip() for k in groq_keys.split(",") if k.strip()]
    print(f"🔍 Encontrada(s) {len(keys)} chave(s) da Groq.\n")
    
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    for i, key in enumerate(keys):
        masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else key
        print(f"🔑 Chave {i+1}/{len(keys)}: {masked_key}")
        
        try:
            client = OpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
            )
            # Faz uma chamada simples para validar
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "oi"}],
                max_tokens=5,
            )
            print("  ✅ Status: FUNCIONANDO")
        except Exception as e:
            err_msg = str(e)
            if "Organization has been restricted" in err_msg or "400" in err_msg:
                print(f"  ❌ Status: FALHOU (Organização Restrita - Conta suspensa ou sem créditos - Erro 400)")
            elif "Rate limit reached" in err_msg or "429" in err_msg:
                print(f"  ⚠️ Status: RATE LIMIT (Muitas requisições - Erro 429)")
            elif "invalid_api_key" in err_msg or "401" in err_msg:
                print(f"  ❌ Status: FALHOU (Chave Inválida - Erro 401)")
            else:
                print(f"  ❌ Status: FALHOU ({err_msg})")
        print_separator()

def test_gemini_keys():
    print("\n" + "="*50)
    print("🚀 TESTANDO CHAVES GEMINI (EMBEDDINGS)")
    print("="*50)
    
    gemini_keys = os.getenv("GOOGLE_API_KEY", "")
    if not gemini_keys:
        print("❌ Nenhuma GOOGLE_API_KEY encontrada no .env")
        return

    keys = [k.strip() for k in gemini_keys.split(",") if k.strip()]
    print(f"🔍 Encontrada(s) {len(keys)} chave(s) do Google Gemini.\n")
    
    for i, key in enumerate(keys):
        masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else key
        print(f"🔑 Chave {i+1}/{len(keys)}: {masked_key}")
        
        try:
            # Teste simples usando a API de embeddings
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={key}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {
                    "parts": [{"text": "teste"}]
                }
            }
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print("  ✅ Status: FUNCIONANDO")
            elif response.status_code == 400:
                print(f"  ❌ Status: FALHOU (Erro 400 - Parâmetros inválidos ou Restrição)")
            elif response.status_code == 429:
                print(f"  ⚠️ Status: RATE LIMIT (Muitas requisições - Erro 429)")
            elif response.status_code == 403:
                print(f"  ❌ Status: FALHOU (Erro 403 - Chave Inválida ou Sem Permissão)")
            else:
                print(f"  ❌ Status: FALHOU ({response.status_code} - {response.text})")
        except Exception as e:
            print(f"  ❌ Status: FALHOU (Exceção: {str(e)})")
        print_separator()

def test_nvidia_keys():
    print("\n" + "="*50)
    print("🚀 TESTANDO CHAVES NVIDIA (FALLBACK LLM)")
    print("="*50)
    
    nvidia_keys = os.getenv("NVIDIA_API_KEY", "")
    if not nvidia_keys:
        print("ℹ️ Nenhuma NVIDIA_API_KEY encontrada no .env (não configurada)")
        return

    keys = [k.strip() for k in nvidia_keys.split(",") if k.strip()]
    print(f"🔍 Encontrada(s) {len(keys)} chave(s) da NVIDIA.\n")
    
    for i, key in enumerate(keys):
        masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else key
        print(f"🔑 Chave {i+1}/{len(keys)}: {masked_key}")
        
        try:
            client = OpenAI(
                api_key=key,
                base_url="https://integrate.api.nvidia.com/v1",
            )
            response = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": "oi"}],
                max_tokens=5,
            )
            print("  ✅ Status: FUNCIONANDO")
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg:
                print(f"  ❌ Status: FALHOU (Chave Inválida - Erro 401)")
            elif "429" in err_msg:
                print(f"  ⚠️ Status: RATE LIMIT (Erro 429)")
            else:
                print(f"  ❌ Status: FALHOU ({err_msg})")
        print_separator()

if __name__ == "__main__":
    test_groq_keys()
    test_nvidia_keys()
    test_gemini_keys()
    print("\n✨ Teste concluído!\n")
