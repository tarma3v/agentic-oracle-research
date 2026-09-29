"""Exa API integration for evidence retrieval"""

import os
import re
from datetime import datetime, timedelta
from typing import Literal, Optional
from exa_py import Exa

from .evidence import Source, EvidencePacket


class ExaOracleRetriever:
    """
    Retrieves web evidence for prediction market questions using Exa API.
    
    Design principle: All LLM agents receive identical evidence packets,
    isolating reasoning capability from retrieval capability.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Exa client.
        
        """
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Exa API key required. Set EXA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.exa = Exa(api_key=self.api_key)
    
    @staticmethod
    def question_to_query(question: str) -> str:
        """
        Convert prediction market question to Exa search query.
        """
        question_clean = question.strip().rstrip("?")
        return f'What is the result of this question "{question_clean}"'
    
    def retrieve(
        self,
        question_id: str,
        question_text: str,
        resolution_criteria: str,
        resolution_date: str,
        retrieval_mode: Literal["full_text", "highlights"] = "highlights",
        num_results: int = 10,
        fulltext_max_characters: int = 4000,
        highlights_max_characters: Optional[int] = 2000,
    ) -> EvidencePacket:
        """
        Retrieve evidence for a prediction market question.
        """
        query = self.question_to_query(question_text)
        
        # Add one day to resolution_date for publication date filtering
        resolution_dt = datetime.fromisoformat(self._normalize_iso8601(resolution_date))
        end_date = (resolution_dt + timedelta(days=1)).isoformat().replace('+00:00', 'Z')
        
        if retrieval_mode == "full_text":
            results = self.exa.search_and_contents(
                query=query,
                type="auto",
                num_results=num_results,
                end_published_date=end_date,
                text={"max_characters": fulltext_max_characters},
            )
        elif retrieval_mode == "highlights":
            highlights_contents: bool | dict
            if highlights_max_characters is None:
                highlights_contents = True
            else:
                highlights_contents = {
                    "query": query,
                    "max_characters": highlights_max_characters,
                }

            results = self.exa.search(
                query=query,
                type="auto",
                num_results=num_results,
                end_published_date=end_date,
                contents={"highlights": highlights_contents},
            )
        else:  # pragma: no cover - protected by CLI choices
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
        
        # Convert Exa results to Source objects
        sources = []
        for result in results.results:
            source = Source(
                title=result.title,
                url=result.url,
                published_date=result.published_date if hasattr(result, "published_date") else None,
                text=result.text if hasattr(result, "text") and result.text else "",
                highlights=list(getattr(result, "highlights", None) or []),
            )
            sources.append(source)
        
        # Package into EvidencePacket
        evidence = EvidencePacket(
            question_id=question_id,
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            resolution_date=resolution_date,
            retrieval_timestamp=datetime.utcnow().isoformat() + "Z",
            query_used=query,
            sources=sources
        )
        
        return evidence

    @staticmethod
    def _normalize_iso8601(value: str) -> str:
        """
        Normalize common ISO-8601 variants into a form accepted by datetime.fromisoformat.
        """
        normalized = value.strip().replace("Z", "+00:00")
        match = re.match(r"^(.*?\.\d+)([+-]\d\d:\d\d)?$", normalized)
        if not match:
            return normalized

        prefix_with_frac = match.group(1)
        tz = match.group(2) or ""
        prefix, frac = prefix_with_frac.split(".", 1)
        frac = (frac + "000000")[:6]
        return f"{prefix}.{frac}{tz}"
    
    def retrieve_and_cache(
        self,
        question_id: str,
        question_text: str,
        resolution_criteria: str,
        resolution_date: str,
        cache_key: Optional[str] = None,
        cache_dir: str = "cache/evidence",
        **kwargs
    ) -> EvidencePacket:
        """
        Retrieve evidence and save to cache.
        
        """
        import json
        from pathlib import Path
        
        # Retrieve evidence
        evidence = self.retrieve(
            question_id=question_id,
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            resolution_date=resolution_date,
            **kwargs
        )
        
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        output_file = cache_path / f"{cache_key or question_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(evidence.to_json())
        
        return evidence
    
    @staticmethod
    def load_from_cache(question_id: str, cache_dir: str = "cache/evidence") -> Optional[EvidencePacket]:
        """
        Load evidence packet from cache.
    
        """
        from pathlib import Path
        
        cache_file = Path(cache_dir) / f"{question_id}.json"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, "r", encoding="utf-8") as f:
            return EvidencePacket.from_json(f.read())
