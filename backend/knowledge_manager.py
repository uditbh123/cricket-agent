import sys
import os
import requests
import time
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

FETCHED_TOPICS_FILE = "./backend/fetched_topics.json"

def load_fetched_topics() -> set:
    if Path(FETCHED_TOPICS_FILE).exists():
        with open(FETCHED_TOPICS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_fetched_topics(topics: set):
    os.makedirs(os.path.dirname(FETCHED_TOPICS_FILE), exist_ok=True)
    with open(FETCHED_TOPICS_FILE, "w") as f:
        json.dump(list(topics), f, indent=2)

import requests

def fetch_wikipedia_article(topic: str) -> Document | None:
    search_url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "cricket-agent/1.0 (educational project; contact@example.com)"
    }

    # Retry up to 3 times with increasing delay
    for attempt in range(3):
        try:
            wait_time = 2 + (attempt * 3)  # 2s, 5s, 8s
            time.sleep(wait_time)

            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "format": "json",
                "srlimit": 3,
                "srprop": "snippet"
            }

            search_response = requests.get(
                search_url,
                params=search_params,
                headers=headers,
                timeout=15
            )

            # Check if response is valid
            if search_response.status_code != 200:
                print(f"  HTTP {search_response.status_code} for '{topic}', retrying...")
                continue

            if not search_response.text.strip():
                print(f"  Empty response for '{topic}', retrying...")
                continue

            search_data = search_response.json()
            results = search_data.get("query", {}).get("search", [])

            if not results:
                print(f"  No results for: {topic}")
                return None

            for result in results:
                page_title = result["title"]
                try:
                    time.sleep(1)
                    content_params = {
                        "action": "query",
                        "titles": page_title,
                        "prop": "extracts",
                        "explaintext": True,
                        "format": "json",
                        "exsectionformat": "plain"
                    }

                    content_response = requests.get(
                        search_url,
                        params=content_params,
                        headers=headers,
                        timeout=15
                    )

                    if not content_response.text.strip():
                        continue

                    content_data = content_response.json()
                    pages = content_data.get("query", {}).get("pages", {})

                    for page_id, page in pages.items():
                        if page_id == "-1":
                            continue
                        content = page.get("extract", "")
                        if len(content) > 500:
                            url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                            print(f"  Matched '{topic}' → '{page_title}'")
                            return Document(
                                page_content=content,
                                metadata={
                                    "source": topic,
                                    "title": page_title,
                                    "url": url
                                }
                            )
                except Exception:
                    continue

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for '{topic}': {e}")
            continue

    print(f"  ✗ All retries failed for: {topic}")
    return None

def add_topics_to_db(topics: list, vectorstore) -> dict:
    fetched_topics = load_fetched_topics()
    new_topics = [t for t in topics if t.lower() not in fetched_topics]
    skipped = [t for t in topics if t.lower() in fetched_topics]

    if not new_topics:
        return {"added": [], "skipped": skipped, "failed": []}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    added = []
    failed = []

    for topic in new_topics:
        doc = fetch_wikipedia_article(topic)
        if doc:
            chunks = splitter.split_documents([doc])
            if chunks:
                vectorstore.add_documents(chunks)
                fetched_topics.add(topic.lower())
                added.append(topic)
                print(f"  ✓ {topic} — {len(chunks)} chunks added")
            else:
                failed.append(topic)
        else:
            failed.append(topic)
            print(f"  ✗ Could not fetch: {topic}")

    save_fetched_topics(fetched_topics)
    return {"added": added, "skipped": skipped, "failed": failed}

def extract_topics_from_question(question: str, llm, history_text: str = "") -> list:
    
    # Include recent conversation so follow-up questions resolve correctly
    history_section = ""
    if history_text:
        history_section = f"""
Recent conversation:
{history_text}

Use the conversation above to understand what topic the user is referring to.
For example if they said "Nepal Premier League" earlier and now ask 
"who won the first season?" — search for "Nepal Premier League" not IPL.
"""

    prompt = f"""You are a Wikipedia search assistant for cricket questions.

Your job: Convert the user's question into 1-3 Wikipedia article titles to search for.
{history_section}
Rules:
- Return ONLY a valid JSON array of strings
- No explanations, no markdown, just the JSON array
- Maximum 3 topics
- Use proper Wikipedia-style titles
- If the question refers to something mentioned in the conversation history, use that

Examples:
"who is the little master" → ["Sachin Tendulkar"]
"who won the first season?" (after discussing Nepal Premier League) → ["Nepal Premier League"]
"how many teams are in it?" (after discussing IPL) → ["Indian Premier League"]
"what is a googly" → ["Googly cricket"]
"fastest bowler ever" → ["Shoaib Akhtar", "Brett Lee"]

Question: "{question}"
JSON array:"""

    try:
        raw = llm.invoke(prompt)
        response = raw.content if hasattr(raw, "content") else str(raw)
        response = response.replace("```json", "").replace("```", "").strip()

        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            topics = json.loads(response[start:end])
            return [t for t in topics if isinstance(t, str)][:3]
    except Exception as e:
        print(f"Topic extraction failed: {e}")

    return []