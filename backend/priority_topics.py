import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from agent import get_vectorstore, _refresh_bm25
from knowledge_manager import add_topics_to_db

PRIORITY_TOPICS = [
    # Core rules & concepts
    "Laws of Cricket",
    "Batting (cricket)",
    "Bowling (cricket)", 
    "Fielding (cricket)",
    "Wicket",
    "Dismissal (cricket)",
    "Leg before wicket",
    "Decision Review System",
    "Duckworth–Lewis–Stern method",
    "Cricket pitch",
    "Cricket ball",
    "Cricket bat",
    
    # Formats
    "Test cricket",
    "One Day International",
    "Twenty20 International",
    "Indian Premier League",
    "Big Bash League",
    "Pakistan Super League",
    "Caribbean Premier League",
    "Nepal Premier League",
    
    # Records & stats
    "List of Test cricket records",
    "List of One Day International cricket records",
    "Cricket World Cup records",
    "Most Test centuries",
    
    # Top players — not in your crawler at all
    "Sachin Tendulkar",
    "Virat Kohli",
    "MS Dhoni",
    "Rohit Sharma",
    "Jasprit Bumrah",
    "Ricky Ponting",
    "Brian Lara",
    "Muttiah Muralitharan",
    "Shane Warne",
    "Jacques Kallis",
    "Kumar Sangakkara",
    "Mahela Jayawardene",
    "AB de Villiers",
    "Steve Smith",
    "David Warner",
    "Pat Cummins",
    "Ben Stokes",
    "Joe Root",
    "Kane Williamson",
    "Babar Azam",
    "Wasim Akram",
    "Waqar Younis",
    "Imran Khan",
    "Kapil Dev",
    "Sunil Gavaskar",
    "Anil Kumble",
    "Rahul Dravid",
    "VVS Laxman",
    "Sourav Ganguly",
    "Glenn McGrath",
    "Adam Gilchrist",
    "Matthew Hayden",
    
    # Tournaments
    "Cricket World Cup",
    "ICC Men's T20 World Cup",
    "ICC Champions Trophy",
    "Asia Cup",
    "World Test Championship",
    "Border-Gavaskar Trophy",
    "The Ashes",
    
    # History
    "History of cricket",
    "Bodyline series",
    "Tied Test",
]

vs = get_vectorstore()
print(f"Force-ingesting {len(PRIORITY_TOPICS)} priority topics...")
result = add_topics_to_db(PRIORITY_TOPICS, vs)
print(f"Added: {len(result['added'])}")
print(f"Skipped (already in DB): {len(result['skipped'])}")
print(f"Failed: {len(result['failed'])}")
if result['failed']:
    print("Failed topics:", result['failed'])