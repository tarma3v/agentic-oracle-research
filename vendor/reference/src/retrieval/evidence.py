"""Data structures for evidence retrieval and caching"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import json


@dataclass
class Source:
    """A single web source retrieved by Exa"""
    
    title: str
    url: str
    published_date: Optional[str] = None
    # Legacy: some packets store full extracted page text.
    text: str = ""
    # Preferred: Exa query-relevant highlights.
    highlights: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        title_preview = self.title[:50] + "..." if len(self.title) > 50 else self.title
        return f"Source(title='{title_preview}', url='{self.url}')"

    def content_for_prompt(self) -> str:
        if self.highlights:
            return "\n".join(f"- {item}" for item in self.highlights)
        return self.text


@dataclass
class EvidencePacket:
    """
    Container for all evidence retrieved for a single question.
    This is the shared input given to all LLM agents.
    """
    
    question_id: str
    question_text: str
    resolution_criteria: str
    resolution_date: str
    retrieval_timestamp: str
    query_used: str
    sources: List[Source] = field(default_factory=list)
    
    def to_json(self) -> str:
        """Serialize to JSON for caching"""
        return json.dumps(
            self.__dict__,
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
            indent=2
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "EvidencePacket":
        """Deserialize from JSON cache"""
        data = json.loads(json_str)
        data["sources"] = [Source(**s) for s in data["sources"]]
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvidencePacket":
        """Create from dictionary (e.g., after JSON load)"""
        sources = [Source(**s) for s in data.get("sources", [])]
        return cls(
            question_id=data["question_id"],
            question_text=data["question_text"],
            resolution_criteria=data["resolution_criteria"],
            resolution_date=data["resolution_date"],
            retrieval_timestamp=data["retrieval_timestamp"],
            query_used=data["query_used"],
            sources=sources
        )
    
    def num_sources(self) -> int:
        """Count of sources retrieved"""
        return len(self.sources)
    
    def total_characters(self) -> int:
        """Total character count across all sources"""
        total = 0
        for source in self.sources:
            if source.highlights:
                total += sum(len(item) for item in source.highlights)
            else:
                total += len(source.text)
        return total
    
    def format_for_prompt(self) -> str:
        """
        Format evidence for LLM consumption.
        Returns a structured text representation.
        """
        lines = [
            f"Question: {self.question_text}",
            f"Resolution Criteria: {self.resolution_criteria}",
            f"Resolution Date: {self.resolution_date}",
            "",
            f"Retrieved {self.num_sources()} sources:",
            ""
        ]
        
        for i, source in enumerate(self.sources, 1):
            lines.append(f"--- Source {i} ---")
            lines.append(f"Title: {source.title}")
            lines.append(f"URL: {source.url}")
            if source.published_date:
                lines.append(f"Published: {source.published_date}")
            content = source.content_for_prompt().strip()
            if content:
                lines.append(f"\n{content}\n")
            else:
                lines.append("\n(No extractable content)\n")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"EvidencePacket(id='{self.question_id}', "
            f"sources={self.num_sources()}, "
            f"chars={self.total_characters()})"
        )
