from duckduckgo_search import DDGS
import json

class WebSearcher:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, max_results=3):
        # Searches the web and returns a clean string of context.
        print(f"[WebSearcher] Searching for: {query}")
        try:
            # SAFETY LIMIT: Only get 3 results
            results = self.ddgs.text(query, max_results=max_results)
            if not results:
                return None
            
            # SAFETY LIMIT: Truncate text to avoid VRAM Crash (Segfault)
            context = "--- WEB SEARCH RESULTS ---\n"
            for res in results:
                title = res.get('title', 'No Title')
                body = res.get('body', '')[:300] # <--- TRUNCATE TO 300 CHARS
                url = res.get('href', 'No URL')
                
                context += f"Title: {title}\n"
                context += f"URL: {url}\n"
                context += f"Content: {body}...\n" # Ellipsis to show cut-off
                context += "---\n"
            
            return context
        except Exception as e:
            print(f"[WebSearcher] Error: {e}")
            return None

    def save_knowledge(self, query, content):
        # Saves the learned content to the local knowledge base.
        # Shorten filename to prevent errors
        clean_query = ''.join(c for c in query if c.isalnum() or c in (' ', '_')).strip()[:20]
        filename = f"knowledge_base/learned_{clean_query.replace(' ', '_')}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Source: Web Search for '{query}'\n\n")
                f.write(content)
            return filename
        except Exception as e:
            print(f"Could not save file: {e}")
            return None
