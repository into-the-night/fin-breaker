"""
Query analyzer for extracting financial entities and determining required agent types.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

from app.backend.models.models import QueryAnalysis, FinancialEntity
from app.backend.models.enums import EntityType, AgentType, Priority
from app.backend.services.synthesis import get_llm_service

logger = logging.getLogger("finbreaker")


class QueryAnalyzer:
    """
    Analyzes financial queries to extract entities and determine required agent types.
    Uses LLM-based entity extraction with rule-based fallbacks.
    """
    
    def __init__(self):
        self.llm_service = get_llm_service()
        
        # Common financial patterns for fallback extraction
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        self.company_keywords = {
            'apple', 'microsoft', 'google', 'amazon', 'tesla', 'meta', 'nvidia',
            'berkshire', 'johnson', 'jpmorgan', 'visa', 'walmart', 'disney'
        }
        self.sector_keywords = {
            'technology', 'healthcare', 'finance', 'energy', 'consumer', 'industrial',
            'materials', 'utilities', 'real estate', 'telecommunications', 'biotech',
            'automotive', 'retail', 'banking', 'insurance', 'pharmaceuticals'
        }
        self.topic_keywords = {
            'earnings', 'revenue', 'profit', 'growth', 'merger', 'acquisition',
            'ipo', 'dividend', 'buyback', 'guidance', 'forecast', 'outlook',
            'competition', 'market share', 'innovation', 'regulation', 'risk'
        }
    
    async def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a financial query and return structured analysis.
        
        Args:
            query: The user's financial query
            
        Returns:
            QueryAnalysis object with extracted entities and required agents
        """
        logger.info(f"Analyzing query: {query}")
        
        query_id = str(uuid.uuid4())
        
        try:
            # Extract entities using LLM
            entities = await self._extract_entities_llm(query)
            
            # Fallback to rule-based extraction if LLM fails or returns insufficient results
            if not entities or len(entities) < 1:
                logger.warning("LLM entity extraction failed or insufficient, using rule-based fallback")
                entities = self._extract_entities_rules(query)
            
            # Determine required agent types based on entities
            required_agents = self._determine_agent_types(entities, query)
            
            # Determine priority based on query characteristics
            priority = self._determine_priority(query, entities)
            
            # Extract context requirements
            context_requirements = self._extract_context_requirements(query, entities)
            
            analysis = QueryAnalysis(
                query_id=query_id,
                original_query=query,
                entities=entities,
                required_agents=required_agents,
                priority=priority,
                context_requirements=context_requirements
            )
            
            logger.info(f"Query analysis completed: {len(entities)} entities, {len(required_agents)} agent types")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Return minimal analysis with rule-based fallback
            entities = self._extract_entities_rules(query)
            required_agents = [AgentType.TOPIC_ANALYSIS] if not entities else self._determine_agent_types(entities, query)
            
            return QueryAnalysis(
                query_id=query_id,
                original_query=query,
                entities=entities,
                required_agents=required_agents,
                priority=Priority.MEDIUM,
                context_requirements={}
            )
    
    async def _extract_entities_llm(self, query: str) -> List[FinancialEntity]:
        """
        Extract financial entities using LLM.
        
        Args:
            query: The user's financial query
            
        Returns:
            List of extracted FinancialEntity objects
        """
        system_prompt = """You are a financial entity extraction specialist. 
        Analyze the given query and extract all relevant financial entities.
        
        For each entity found, provide:
        1. entity_type: one of [company, ticker, topic, sector, currency, commodity]
        2. value: the actual entity text
        3. confidence: float between 0.0 and 1.0
        4. metadata: any additional context
        
        Return your response as a JSON array of entities. Example:
        [
            {"entity_type": "company", "value": "Apple Inc", "confidence": 0.95, "metadata": {"industry": "technology"}},
            {"entity_type": "ticker", "value": "AAPL", "confidence": 0.98, "metadata": {}},
            {"entity_type": "topic", "value": "earnings", "confidence": 0.85, "metadata": {"context": "quarterly results"}}
        ]
        
        If no clear financial entities are found, return an empty array []."""
        
        try:
            response = await self.llm_service.generate(
                messages=[{
                    "role": "user",
                    "content": f"Extract financial entities from this query: {query}"
                }],
                model='gemini-2.0-flash-001',
                system=system_prompt,
                temperature=0.1
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse JSON response (handle markdown code blocks)
            import json
            import re
            
            # Remove markdown code blocks if present
            json_text = response_text.strip()
            if json_text.startswith('```json'):
                json_text = re.sub(r'^```json\s*', '', json_text)
                json_text = re.sub(r'\s*```$', '', json_text)
            elif json_text.startswith('```'):
                json_text = re.sub(r'^```\s*', '', json_text)
                json_text = re.sub(r'\s*```$', '', json_text)
            
            try:
                entities_data = json.loads(json_text)
                entities = []
                
                for entity_data in entities_data:
                    try:
                        entity = FinancialEntity(
                            entity_type=EntityType(entity_data['entity_type']),
                            value=entity_data['value'],
                            confidence=float(entity_data['confidence']),
                            metadata=entity_data.get('metadata', {})
                        )
                        entities.append(entity)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Invalid entity data: {entity_data}, error: {e}")
                        continue
                
                return entities
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {response_text}, error: {e}")
                return []
                
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {str(e)}")
            return []
    
    def _extract_entities_rules(self, query: str) -> List[FinancialEntity]:
        """
        Extract entities using rule-based patterns as fallback.
        
        Args:
            query: The user's financial query
            
        Returns:
            List of extracted FinancialEntity objects
        """
        entities = []
        query_lower = query.lower()
        
        # Extract tickers (uppercase 1-5 letter words)
        tickers = self.ticker_pattern.findall(query)
        for ticker in tickers:
            if len(ticker) <= 5 and ticker.isalpha():
                entities.append(FinancialEntity(
                    entity_type=EntityType.TICKER,
                    value=ticker,
                    confidence=0.7,
                    metadata={"extraction_method": "regex"}
                ))
        
        # Extract companies by keyword matching
        for company in self.company_keywords:
            if company in query_lower:
                entities.append(FinancialEntity(
                    entity_type=EntityType.COMPANY,
                    value=company.title(),
                    confidence=0.6,
                    metadata={"extraction_method": "keyword"}
                ))
        
        # Extract sectors
        for sector in self.sector_keywords:
            if sector in query_lower:
                entities.append(FinancialEntity(
                    entity_type=EntityType.SECTOR,
                    value=sector.title(),
                    confidence=0.6,
                    metadata={"extraction_method": "keyword"}
                ))
        
        # Extract topics
        for topic in self.topic_keywords:
            if topic in query_lower:
                entities.append(FinancialEntity(
                    entity_type=EntityType.TOPIC,
                    value=topic.title(),
                    confidence=0.6,
                    metadata={"extraction_method": "keyword"}
                ))
        
        # If no entities found, create a general topic entity
        if not entities:
            entities.append(FinancialEntity(
                entity_type=EntityType.TOPIC,
                value="General Financial Query",
                confidence=0.5,
                metadata={"extraction_method": "fallback"}
            ))
        
        return entities
    
    def _determine_agent_types(self, entities: List[FinancialEntity], query: str) -> List[AgentType]:
        """
        Determine required agent types based on extracted entities and query content.
        
        Args:
            entities: List of extracted financial entities
            query: Original query text
            
        Returns:
            List of required AgentType values
        """
        required_agents = set()
        query_lower = query.lower()
        
        # Analyze entities to determine agent requirements
        has_company_or_ticker = any(e.entity_type in [EntityType.COMPANY, EntityType.TICKER] for e in entities)
        has_sector = any(e.entity_type == EntityType.SECTOR for e in entities)
        has_topic = any(e.entity_type == EntityType.TOPIC for e in entities)
        
        # Market data agent for price, volume, technical analysis
        market_keywords = ['price', 'stock', 'volume', 'chart', 'technical', 'moving average', 'rsi', 'macd']
        if has_company_or_ticker or any(keyword in query_lower for keyword in market_keywords):
            required_agents.add(AgentType.MARKET_DATA)
        
        # Company research agent for company-specific information
        company_keywords = ['earnings', 'revenue', 'financial', 'balance sheet', 'income statement', 'cash flow']
        if has_company_or_ticker or any(keyword in query_lower for keyword in company_keywords):
            required_agents.add(AgentType.COMPANY_RESEARCH)
        
        # Topic analysis agent for thematic or sector analysis
        topic_keywords = ['trend', 'outlook', 'forecast', 'industry', 'sector', 'market', 'analysis']
        if has_topic or has_sector or any(keyword in query_lower for keyword in topic_keywords):
            required_agents.add(AgentType.TOPIC_ANALYSIS)
        
        # Risk analysis agent for risk-related queries
        risk_keywords = ['risk', 'volatility', 'beta', 'correlation', 'var', 'exposure', 'hedge']
        if any(keyword in query_lower for keyword in risk_keywords):
            required_agents.add(AgentType.RISK_ANALYSIS)
        
        # Ensure at least one agent is required
        if not required_agents:
            required_agents.add(AgentType.TOPIC_ANALYSIS)
        
        return list(required_agents)
    
    def _determine_priority(self, query: str, entities: List[FinancialEntity]) -> Priority:
        """
        Determine query priority based on content and entities.
        
        Args:
            query: Original query text
            entities: List of extracted entities
            
        Returns:
            Priority level
        """
        query_lower = query.lower()
        
        # High priority keywords
        urgent_keywords = ['urgent', 'asap', 'immediately', 'breaking', 'alert']
        high_keywords = ['earnings', 'announcement', 'merger', 'acquisition', 'ipo']
        
        if any(keyword in query_lower for keyword in urgent_keywords):
            return Priority.URGENT
        elif any(keyword in query_lower for keyword in high_keywords):
            return Priority.HIGH
        elif len(entities) > 3:  # Complex queries with many entities
            return Priority.HIGH
        else:
            return Priority.MEDIUM
    
    def _extract_context_requirements(self, query: str, entities: List[FinancialEntity]) -> Dict[str, Any]:
        """
        Extract context requirements for agent execution.
        
        Args:
            query: Original query text
            entities: List of extracted entities
            
        Returns:
            Dictionary of context requirements
        """
        requirements = {
            "time_horizon": self._extract_time_horizon(query),
            "data_types": self._extract_data_types(query),
            "analysis_depth": self._extract_analysis_depth(query),
            "comparison_required": self._requires_comparison(query),
            "entities_by_type": self._group_entities_by_type(entities)
        }
        
        return requirements
    
    def _extract_time_horizon(self, query: str) -> str:
        """Extract time horizon from query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['daily', 'today', 'current']):
            return 'daily'
        elif any(word in query_lower for word in ['weekly', 'week']):
            return 'weekly'
        elif any(word in query_lower for word in ['monthly', 'month']):
            return 'monthly'
        elif any(word in query_lower for word in ['quarterly', 'quarter', 'q1', 'q2', 'q3', 'q4']):
            return 'quarterly'
        elif any(word in query_lower for word in ['yearly', 'annual', 'year']):
            return 'yearly'
        else:
            return 'default'
    
    def _extract_data_types(self, query: str) -> List[str]:
        """Extract required data types from query."""
        query_lower = query.lower()
        data_types = []
        
        type_mapping = {
            'price': ['price', 'stock price', 'share price'],
            'volume': ['volume', 'trading volume'],
            'financial': ['earnings', 'revenue', 'profit', 'financial'],
            'news': ['news', 'announcement', 'press release'],
            'analyst': ['analyst', 'recommendation', 'rating'],
            'technical': ['technical', 'chart', 'indicator']
        }
        
        for data_type, keywords in type_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                data_types.append(data_type)
        
        return data_types if data_types else ['general']
    
    def _extract_analysis_depth(self, query: str) -> str:
        """Extract required analysis depth."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['detailed', 'comprehensive', 'thorough', 'deep']):
            return 'detailed'
        elif any(word in query_lower for word in ['brief', 'summary', 'quick', 'overview']):
            return 'brief'
        else:
            return 'standard'
    
    def _requires_comparison(self, query: str) -> bool:
        """Check if query requires comparison analysis."""
        query_lower = query.lower()
        comparison_keywords = ['compare', 'vs', 'versus', 'against', 'better', 'worse', 'relative']
        return any(keyword in query_lower for keyword in comparison_keywords)
    
    def _group_entities_by_type(self, entities: List[FinancialEntity]) -> Dict[str, List[str]]:
        """Group entities by their type."""
        grouped = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity.value)
        return grouped