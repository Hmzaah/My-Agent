from duckduckgo_search import DDGS
import time

class WebSearcher:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, max_results=3):
        print(f"[WebSearcher] 🔎 Searching: {query}")
        
        # Try 3 different backends to bypass blocks
        backends = ['api', 'html', 'lite']
        
        for backend in backends:
            try:
                print(f"[WebSearcher] Trying backend: '{backend}'...")
                results = self.ddgs.text(query, max_results=max_results, backend=backend)
                
                if results:
                    context = "--- WEB SEARCH RESULTS ---\n"
                    for res in results:
                        body = res.get('body', '')[:1500] 
                        context += f"URL: {res.get('href', 'N/A')}\n"
                        context += f"Content: {body}\n---\n"
                    
                    print(f"[WebSearcher] ✅ Success with '{backend}'!")
                    return context
            except Exception as e:
                print(f"[WebSearcher] ⚠️ Backend '{backend}' failed: {e}")
                time.sleep(1) # Wait a bit before retrying
        
        print("[WebSearcher] ❌ All backends failed.")
        return None

    def save_knowledge(self, query, content):
        clean_query = ''.join(c for c in query if c.isalnum() or c in (' ', '_')).strip()[:30]
        filename = f"knowledge_base/learned_{clean_query.replace(' ', '_')}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Source: Web Search for '{query}'\n\n{content}")
            return filename
        except:
            return None
