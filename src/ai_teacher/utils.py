import re

def convert_to_json(crew_result: str) -> list[dict]:
    """Convert CrewAI numbered output to list of dicts."""
    pattern = r"\d+\.\s+\*\*(.*?)\*\*:\s*(.*?)(?=\n\d+\.\s+\*\*|\Z)"
    matches = re.findall(pattern, crew_result, re.DOTALL)
    return [
        {"context_title": title.strip(), "context": context.strip()}
        for title, context in matches
    ]