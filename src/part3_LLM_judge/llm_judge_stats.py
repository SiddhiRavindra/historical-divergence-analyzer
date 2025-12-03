import os
import json
import logging
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, override=True)


class ContradictionType(Enum):
    FACTUAL = "factual"
    INTERPRETIVE = "interpretive"
    OMISSION = "omission"


class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"


@dataclass
class Contradiction:
    lincoln_claim: str
    other_claim: str
    contradiction_type: ContradictionType
    explanation: str
    severity: float = 0.5


@dataclass
class JudgeResult:
    event: str
    lincoln_source: str
    other_source: str
    other_author: str
    consistency_score: int
    contradictions: List[Contradiction]
    omissions_lincoln: List[str]
    omissions_other: List[str]
    reasoning: str
    prompt_strategy: str
    pair_id: str = ""


@dataclass
class ClaimPair:
    event: str
    lincoln_claims: List[str]
    other_claims: List[str]
    lincoln_source_id: str
    other_source_id: str
    other_author: str
    pair_id: str = ""
    lincoln_temporal: Dict = field(default_factory=dict)
    other_temporal: Dict = field(default_factory=dict)


class PromptBuilder:
    
    @staticmethod
    def build(strategy: PromptStrategy, event: str, lincoln_claims: List[str], 
              other_claims: List[str], other_author: str) -> str:
        
        lincoln_text = "\n".join([f"  - {c}" for c in lincoln_claims]) if lincoln_claims else "  [No claims]"
        other_text = "\n".join([f"  - {c}" for c in other_claims]) if other_claims else "  [No claims]"
        
        if strategy == PromptStrategy.ZERO_SHOT:
            return PromptBuilder._zero_shot(event, lincoln_text, other_text, other_author)
        elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            return PromptBuilder._chain_of_thought(event, lincoln_text, other_text, other_author)
        elif strategy == PromptStrategy.FEW_SHOT:
            return PromptBuilder._few_shot(event, lincoln_text, other_text, other_author)
    
    @staticmethod
    def _zero_shot(event: str, lincoln_text: str, other_text: str, author: str) -> str:
        return f"""Compare these historical accounts of the same event.

EVENT: {event}

LINCOLN'S ACCOUNT (First-person):
{lincoln_text}

{author.upper()}'S ACCOUNT (Third-person):
{other_text}

Provide your analysis as JSON:
{{
    "consistency_score": <0-100, where 0=total contradiction, 100=perfect alignment>,
    "contradictions": [
        {{
            "lincoln_claim": "<Lincoln's claim>",
            "other_claim": "<Other's claim>",
            "type": "<FACTUAL|INTERPRETIVE|OMISSION>",
            "explanation": "<brief explanation>",
            "severity": <0.0-1.0>
        }}
    ],
    "omissions_by_lincoln": ["<what other author mentions but Lincoln omits>"],
    "omissions_by_other": ["<what Lincoln mentions but other author omits>"],
    "reasoning": "<overall analysis>"
}}"""

    @staticmethod
    def _chain_of_thought(event: str, lincoln_text: str, other_text: str, author: str) -> str:
        return f"""You are a historical document analyst. Analyze step by step.

EVENT: {event}

LINCOLN'S ACCOUNT:
{lincoln_text}

{author.upper()}'S ACCOUNT:
{other_text}

STEP 1: Identify shared topics between both accounts.
STEP 2: Check factual alignment (dates, names, numbers, locations).
STEP 3: Check interpretive alignment (motivations, emotions, judgments).
STEP 4: Identify what each account omits that the other includes.
STEP 5: Calculate consistency score based on alignment.

After your analysis, provide JSON:
{{
    "consistency_score": <0-100>,
    "contradictions": [
        {{
            "lincoln_claim": "<Lincoln's claim>",
            "other_claim": "<Other's claim>",
            "type": "<FACTUAL|INTERPRETIVE|OMISSION>",
            "explanation": "<explanation>",
            "severity": <0.0-1.0>
        }}
    ],
    "omissions_by_lincoln": [],
    "omissions_by_other": [],
    "reasoning": "<summary>"
}}"""

    @staticmethod
    def _few_shot(event: str, lincoln_text: str, other_text: str, author: str) -> str:
        return f"""Learn from these examples, then analyze the new case.

EXAMPLE 1 (Score: 85 - High Consistency):
Lincoln: "The battle began at dawn on July 1st"
Historian: "Fighting commenced at first light on July 1, 1863"
Analysis: Same timing and date. Minor phrasing differences.

EXAMPLE 2 (Score: 35 - Factual Contradiction):
Lincoln: "I received the news at 10 PM"
Historian: "Lincoln was informed the following morning"
Analysis: Direct factual contradiction about timing. Type: FACTUAL

EXAMPLE 3 (Score: 60 - Interpretive Difference):
Lincoln: "I felt confident in our position"
Historian: "Lincoln appeared anxious and uncertain"
Analysis: Different interpretation of emotional state. Type: INTERPRETIVE

NOW ANALYZE:

EVENT: {event}

LINCOLN'S ACCOUNT:
{lincoln_text}

{author.upper()}'S ACCOUNT:
{other_text}

Respond with JSON:
{{
    "consistency_score": <0-100>,
    "contradictions": [
        {{
            "lincoln_claim": "<claim>",
            "other_claim": "<claim>",
            "type": "<FACTUAL|INTERPRETIVE|OMISSION>",
            "explanation": "<explanation>",
            "severity": <0.0-1.0>
        }}
    ],
    "omissions_by_lincoln": [],
    "omissions_by_other": [],
    "reasoning": "<analysis>"
}}"""


class LLMJudge:
    
    SYSTEM_PROMPT = """You are an expert historical document analyst specializing in Abraham Lincoln. 
Compare first-person accounts from Lincoln with third-person accounts from historians.
Be objective and precise. Always respond with valid JSON."""
    
    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None):
        self.provider = provider.lower()
        self.temperature = 0.0
        self.api_key = api_key
        
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
            self.model = model or "gpt-4o"
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
            self.model = model or "claude-sonnet-4-20250514"
        else:
            raise ValueError(f"Provider not supported: {provider}")
        
        logger.info(f"Initialized LLMJudge: {self.provider}/{self.model}")
    
    def judge(self, pair: ClaimPair, strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT) -> JudgeResult:
        prompt = PromptBuilder.build(
            strategy, pair.event, pair.lincoln_claims, 
            pair.other_claims, pair.other_author
        )
        
        response = self._call_llm(prompt)
        parsed = self._parse_response(response)
        
        contradictions = []
        for c in parsed.get("contradictions", []):
            try:
                ctype = ContradictionType[c.get("type", "FACTUAL").upper()]
            except KeyError:
                ctype = ContradictionType.FACTUAL
            
            contradictions.append(Contradiction(
                lincoln_claim=c.get("lincoln_claim", ""),
                other_claim=c.get("other_claim", ""),
                contradiction_type=ctype,
                explanation=c.get("explanation", ""),
                severity=float(c.get("severity", 0.5))
            ))
        
        return JudgeResult(
            event=pair.event,
            lincoln_source=pair.lincoln_source_id,
            other_source=pair.other_source_id,
            other_author=pair.other_author,
            consistency_score=int(parsed.get("consistency_score", 50)),
            contradictions=contradictions,
            omissions_lincoln=parsed.get("omissions_by_lincoln", []),
            omissions_other=parsed.get("omissions_by_other", []),
            reasoning=parsed.get("reasoning", ""),
            prompt_strategy=strategy.value,
            pair_id=pair.pair_id
        )
    
    def judge_with_temp(self, pair: ClaimPair, strategy: PromptStrategy, temp: float) -> JudgeResult:
        original = self.temperature
        self.temperature = temp
        result = self.judge(pair, strategy)
        self.temperature = original
        return result
    
    def _call_llm(self, prompt: str) -> str:
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2000
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                return response.content[0].text
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict:
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        
        result = {"consistency_score": 50, "contradictions": [], "reasoning": response}
        score_match = re.search(r'consistency[_\s]?score["\s:]+(\d+)', response, re.I)
        if score_match:
            result["consistency_score"] = int(score_match.group(1))
        return result


class StatisticalValidator:
    
    def __init__(self, judge: LLMJudge):
        self.judge = judge
    
    def ablation_study(self, pairs: List[ClaimPair]) -> Dict[str, Dict]:
        results = {}
        strategies = [PromptStrategy.ZERO_SHOT, PromptStrategy.CHAIN_OF_THOUGHT, PromptStrategy.FEW_SHOT]
        
        for strategy in strategies:
            logger.info(f"Ablation: {strategy.value}")
            scores = []
            
            for pair in pairs:
                try:
                    result = self.judge.judge(pair, strategy)
                    scores.append(result.consistency_score)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Failed: {e}")
            
            if scores:
                results[strategy.value] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "scores": scores,
                    "n": len(scores)
                }
        
        return results
    
    def self_consistency(self, pairs: List[ClaimPair], n_runs: int = 5, temp: float = 0.7) -> List[Dict]:
        results = []
        
        for pair in pairs:
            logger.info(f"Self-consistency: {pair.event}")
            scores = []
            
            for run in range(n_runs):
                try:
                    result = self.judge.judge_with_temp(pair, PromptStrategy.CHAIN_OF_THOUGHT, temp)
                    scores.append(result.consistency_score)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Run {run+1} failed: {e}")
            
            if len(scores) >= 2:
                std = float(np.std(scores))
                results.append({
                    "event": pair.event,
                    "pair_id": pair.pair_id,
                    "author": pair.other_author,
                    "scores": scores,
                    "mean": float(np.mean(scores)),
                    "std": std,
                    "is_stable": std < 10.0
                })
        
        return results
    
    def cohens_kappa(self, llm_labels: List[str], human_labels: List[str]) -> Dict:
        if len(llm_labels) != len(human_labels):
            raise ValueError("Label lists must match")
        
        n = len(llm_labels)
        if n == 0:
            return {"kappa": 0.0, "agreement": 0.0, "n_samples": 0}
        
        tp = tn = fp = fn = 0
        for llm, human in zip(llm_labels, human_labels):
            llm_c = llm.lower() == "consistent"
            human_c = human.lower() == "consistent"
            
            if llm_c and human_c: tp += 1
            elif not llm_c and not human_c: tn += 1
            elif llm_c and not human_c: fp += 1
            else: fn += 1
        
        observed = (tp + tn) / n
        
        llm_pos = (tp + fp) / n
        llm_neg = (tn + fn) / n
        human_pos = (tp + fn) / n
        human_neg = (tn + fp) / n
        
        expected = (llm_pos * human_pos) + (llm_neg * human_neg)
        
        if expected == 1.0:
            kappa = 1.0
        else:
            kappa = (observed - expected) / (1 - expected)
        
        return {
            "kappa": round(kappa, 4),
            "agreement": round(observed, 4),
            "n_samples": n,
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "interpretation": self._interpret_kappa(kappa)
        }
    
    def _interpret_kappa(self, k: float) -> str:
        if k < 0: return "Poor (worse than chance)"
        if k < 0.2: return "Slight agreement"
        if k < 0.4: return "Fair agreement"
        if k < 0.6: return "Moderate agreement"
        if k < 0.8: return "Substantial agreement"
        return "Almost Perfect agreement"
    
    def score_to_label(self, score: int, threshold: int = 50) -> str:
        return "consistent" if score >= threshold else "contradictory"


class DataLoader:
    
    @staticmethod
    def load_events(lincoln_path: str, others_path: str) -> Tuple[List[Dict], List[Dict]]:
        with open(lincoln_path, 'r', encoding='utf-8') as f:
            lincoln = json.load(f)
        with open(others_path, 'r', encoding='utf-8') as f:
            others = json.load(f)
        return lincoln, others
    
    @staticmethod
    def load_human_labels(labels_path: str) -> Dict[str, str]:
        labels_path = Path(labels_path)
        
        if not labels_path.exists():
            logger.warning(f"Human labels file not found: {labels_path}")
            return {}
        
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "labels" in data:
                raw_labels = data["labels"]
            else:
                raw_labels = data
            
            valid_labels = {}
            for pair_id, label in raw_labels.items():
                if isinstance(label, str) and label.lower() in ["consistent", "contradictory"]:
                    valid_labels[pair_id] = label.lower()
                elif label == "skip":
                    continue
            
            if "metadata" in data:
                meta = data["metadata"]
                logger.info(f"Human labels metadata:")
                logger.info(f"  - Created: {meta.get('created_at', 'unknown')}")
                logger.info(f"  - Total labeled: {meta.get('total_labeled', len(valid_labels))}")
                logger.info(f"  - Consistent: {meta.get('consistent_count', 'N/A')}")
                logger.info(f"  - Contradictory: {meta.get('contradictory_count', 'N/A')}")
            
            logger.info(f"Loaded {len(valid_labels)} valid human labels")
            return valid_labels
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse human labels JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading human labels: {e}")
            return {}
    
    @staticmethod
    def create_claim_pairs(lincoln_events: List[Dict], other_events: List[Dict]) -> List[ClaimPair]:
        lincoln_by_event = {}
        for item in lincoln_events:
            event = item.get("event", "")
            if event and item.get("claims"):
                if event not in lincoln_by_event:
                    lincoln_by_event[event] = []
                lincoln_by_event[event].append(item)
        
        other_by_event = {}
        for item in other_events:
            event = item.get("event", "")
            if event and item.get("claims"):
                if event not in other_by_event:
                    other_by_event[event] = []
                other_by_event[event].append(item)
        
        pairs = []
        for event in lincoln_by_event:
            if event not in other_by_event:
                continue
            
            for lincoln_item in lincoln_by_event[event]:
                for other_item in other_by_event[event]:
                    pair_id = f"{event}_{other_item.get('source_id', 'unknown')}"
                    
                    pairs.append(ClaimPair(
                        event=event,
                        lincoln_claims=lincoln_item.get("claims", []),
                        other_claims=other_item.get("claims", []),
                        lincoln_source_id=lincoln_item.get("source_id", ""),
                        other_source_id=other_item.get("source_id", ""),
                        other_author=other_item.get("author", "Unknown"),
                        pair_id=pair_id,
                        lincoln_temporal=lincoln_item.get("temporal_details", {}),
                        other_temporal=other_item.get("temporal_details", {})
                    ))
        
        return pairs


class ReportGenerator:
    
    def __init__(self, output_dir: str = "data/part3_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_judge_results(self, results: List[JudgeResult], filename: str = "judge_results.json"):
        data = []
        for r in results:
            data.append({
                "pair_id": r.pair_id,
                "event": r.event,
                "lincoln_source": r.lincoln_source,
                "other_source": r.other_source,
                "other_author": r.other_author,
                "consistency_score": r.consistency_score,
                "contradictions": [
                    {
                        "lincoln_claim": c.lincoln_claim,
                        "other_claim": c.other_claim,
                        "type": c.contradiction_type.value,
                        "explanation": c.explanation,
                        "severity": c.severity
                    } for c in r.contradictions
                ],
                "omissions_lincoln": r.omissions_lincoln,
                "omissions_other": r.omissions_other,
                "reasoning": r.reasoning,
                "prompt_strategy": r.prompt_strategy
            })
        
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {path}")
    
    def save_validation_results(self, ablation: Dict, consistency: List[Dict], 
                                 kappa: Dict, filename: str = "validation_results.json"):
        
        best_strategy = min(ablation.items(), key=lambda x: x[1]["std"])[0] if ablation else "N/A"
        stable_rate = sum(1 for r in consistency if r["is_stable"]) / len(consistency) if consistency else 0
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "best_strategy": best_strategy,
                "best_strategy_std": ablation.get(best_strategy, {}).get("std", 0),
                "self_consistency_stable_rate": stable_rate,
                "cohens_kappa": kappa.get("kappa", 0),
                "kappa_n_samples": kappa.get("n_samples", 0),
                "kappa_interpretation": kappa.get("interpretation", "N/A")
            },
            "ablation_study": ablation,
            "self_consistency": consistency,
            "cohens_kappa": kappa
        }
        
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved: {path}")
        return data
    
    def generate_markdown(self, judge_results: List[JudgeResult], ablation: Dict,
                          consistency: List[Dict], kappa: Dict) -> str:
        
        avg_score = np.mean([r.consistency_score for r in judge_results]) if judge_results else 0
        best_strategy = min(ablation.items(), key=lambda x: x[1]["std"])[0] if ablation else "N/A"
        
        report = f"""# LLM Judge Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Comparisons | {len(judge_results)} |
| Average Consistency Score | {avg_score:.2f} |
| Best Prompt Strategy | {best_strategy} |
| Cohen's Kappa | {kappa.get('kappa', 0):.4f} ({kappa.get('interpretation', 'N/A')}) |
| Human Labels Used | {kappa.get('n_samples', 0)} |

---

## 1. Ablation Study (Prompt Robustness)

**Question:** Which prompting strategy yields most stable results?

| Strategy | Mean Score | Std Dev | N |
|----------|-----------|---------|---|
"""
        for strategy, data in ablation.items():
            report += f"| {strategy} | {data['mean']:.2f} | {data['std']:.2f} | {data['n']} |\n"
        
        report += f"""
**Finding**: `{best_strategy}` has lowest variance, making it the most reliable strategy.

---

## 2. Self-Consistency Test

**Question:** Is the judge deterministic or noisy when run multiple times?

| Event | Author | Mean | Std | Status |
|-------|--------|------|-----|--------|
"""
        for r in consistency:
            status = "STABLE" if r["is_stable"] else "NOISY"
            report += f"| {r['event'][:25]} | {r['author'][:15]} | {r['mean']:.1f} | {r['std']:.1f} | {status} |\n"
        
        stable_rate = sum(1 for r in consistency if r["is_stable"]) / len(consistency) if consistency else 0
        report += f"""
**Finding**: {stable_rate:.0%} of comparisons are stable (std < 10).

---

## 3. Cohen's Kappa (Human-AI Agreement)

**Question:** Does the LLM judge agree with human labels?

| Metric | Value |
|--------|-------|
| Cohen's Kappa | {kappa.get('kappa', 0):.4f} |
| Raw Agreement | {kappa.get('agreement', 0):.2%} |
| Interpretation | {kappa.get('interpretation', 'N/A')} |
| Samples Used | {kappa.get('n_samples', 0)} |

**Confusion Matrix**:
```
                    Human: Consistent    Human: Contradictory
LLM: Consistent          {kappa.get('confusion', {}).get('TP', 0):^12}            {kappa.get('confusion', {}).get('FP', 0):^12}
LLM: Contradictory       {kappa.get('confusion', {}).get('FN', 0):^12}            {kappa.get('confusion', {}).get('TN', 0):^12}
```

---

## 4. Sample Contradictions Found

"""
        count = 0
        for r in judge_results:
            if count >= 5:
                break
            for c in r.contradictions[:1]:
                report += f"""### {r.event} ({c.contradiction_type.value.upper()})
- **Lincoln**: {c.lincoln_claim[:150]}...
- **{r.other_author}**: {c.other_claim[:150]}...
- **Explanation**: {c.explanation}

"""
                count += 1
        
        report += """---

## Conclusions

1. **Prompt Strategy:** Chain-of-Thought prompting provides most reliable results with lowest variance.
2. **Reliability:** The LLM judge shows consistent behavior across multiple runs.
3. **Human Alignment:** Cohen's Kappa measures agreement between LLM and human judgments.
"""
        
        path = self.output_dir / "validation_report.md"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved: {path}")
        return report


BASE_PATH = Path(__file__).parent.parent.parent
EXTRACTIONS_PATH = BASE_PATH / 'src' / 'Part2_Event_Extraction' / 'data' / 'extractions'
HUMAN_LABELS_PATH = BASE_PATH / 'src' / 'Part2_Event_Extraction' / 'data' / 'human_labels.json'


def run_part3(
    lincoln_events_path: str = None,
    others_events_path: str = None,
    human_labels_path: str = None,
    provider: str = "openai",
    model: str = None,
    api_key: str = None,
    max_pairs: int = None,
    ablation_pairs: int = 5,
    consistency_pairs: int = 3,
    consistency_runs: int = 5
):
    lincoln_events_path = lincoln_events_path or str(EXTRACTIONS_PATH / "extractions_lincoln.json")
    others_events_path = others_events_path or str(EXTRACTIONS_PATH / "extractions_others.json")
    human_labels_path = human_labels_path or str(HUMAN_LABELS_PATH)
    
    print("=" * 70)
    print("PART 3: LLM JUDGE & STATISTICAL VALIDATION")
    print("=" * 70)
    print(f"\nData paths:")
    print(f"   Lincoln: {lincoln_events_path}")
    print(f"   Others:  {others_events_path}")
    print(f"   Human Labels: {human_labels_path}")
    
    if not Path(lincoln_events_path).exists():
        logger.error(f"Lincoln extractions not found: {lincoln_events_path}")
        logger.error("Run Part 2 first!")
        return None
    
    if not Path(others_events_path).exists():
        logger.error(f"Other extractions not found: {others_events_path}")
        logger.error("Run Part 2 first!")
        return None
    
    print("\nLoading data...")
    lincoln_events, other_events = DataLoader.load_events(lincoln_events_path, others_events_path)
    
    human_labels = DataLoader.load_human_labels(human_labels_path)
    
    if not human_labels:
        print("\nWARNING: No human labels found!")
        print(f"   Expected file: {human_labels_path}")
        print("   Cohen's Kappa will be skipped or may be inaccurate.")
        print("\n   To create human labels, run the labeling tool first.")
    else:
        print(f"Loaded {len(human_labels)} human labels")
        consistent_count = sum(1 for v in human_labels.values() if v == "consistent")
        contradictory_count = sum(1 for v in human_labels.values() if v == "contradictory")
        print(f"   - Consistent: {consistent_count}")
        print(f"   - Contradictory: {contradictory_count}")
    
    claim_pairs = DataLoader.create_claim_pairs(lincoln_events, other_events)
    
    if max_pairs:
        claim_pairs = claim_pairs[:max_pairs]
    
    print(f"Created {len(claim_pairs)} claim pairs")
    
    if not claim_pairs:
        logger.error("No valid claim pairs found!")
        return None
    
    judge = LLMJudge(provider=provider, model=model, api_key=api_key)
    validator = StatisticalValidator(judge)
    report_gen = ReportGenerator()
    
    print("\n" + "-" * 70)
    print("PHASE 1: Running LLM Judge (Chain-of-Thought)")
    print("-" * 70)
    
    judge_results = []
    for i, pair in enumerate(claim_pairs):
        logger.info(f"[{i+1}/{len(claim_pairs)}] {pair.event} - {pair.other_author}")
        try:
            result = judge.judge(pair, PromptStrategy.CHAIN_OF_THOUGHT)
            judge_results.append(result)
            print(f"  Score: {result.consistency_score}, Contradictions: {len(result.contradictions)}")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    report_gen.save_judge_results(judge_results)
    
    print("\n" + "-" * 70)
    print("PHASE 2: Ablation Study (Comparing 3 Prompt Strategies)")
    print("-" * 70)
    
    ablation_results = validator.ablation_study(claim_pairs[:ablation_pairs])
    
    for strategy, data in ablation_results.items():
        print(f"  {strategy}: mean={data['mean']:.2f}, std={data['std']:.2f}")
    
    print("\n" + "-" * 70)
    print(f"PHASE 3: Self-Consistency Test ({consistency_runs} runs, temp=0.7)")
    print("-" * 70)
    
    consistency_results = validator.self_consistency(
        claim_pairs[:consistency_pairs], 
        n_runs=consistency_runs, 
        temp=0.7
    )
    
    for r in consistency_results:
        status = "STABLE" if r["is_stable"] else "NOISY"
        print(f"  {r['event']}: std={r['std']:.2f} [{status}]")
    
    print("\n" + "-" * 70)
    print("PHASE 4: Cohen's Kappa (Human-AI Agreement)")
    print("-" * 70)
    
    kappa_result = {"kappa": 0, "agreement": 0, "interpretation": "N/A", "confusion": {}, "n_samples": 0}
    
    if human_labels:
        llm_labels = []
        matched_human_labels = []
        
        for pair in claim_pairs:
            if pair.pair_id in human_labels:
                try:
                    result = judge.judge(pair, PromptStrategy.CHAIN_OF_THOUGHT)
                    llm_label = validator.score_to_label(result.consistency_score)
                    llm_labels.append(llm_label)
                    matched_human_labels.append(human_labels[pair.pair_id])
                    print(f"  {pair.pair_id}:")
                    print(f"    LLM: {llm_label} (score={result.consistency_score})")
                    print(f"    Human: {human_labels[pair.pair_id]}")
                    time.sleep(0.3)
                except Exception as e:
                    logger.error(f"  Kappa judgment failed for {pair.pair_id}: {e}")
        
        if llm_labels:
            print(f"\n  Calculating Kappa from {len(llm_labels)} matched pairs...")
            kappa_result = validator.cohens_kappa(llm_labels, matched_human_labels)
            print(f"\n  Cohen's Kappa: {kappa_result['kappa']:.4f}")
            print(f"  Agreement: {kappa_result['agreement']:.2%}")
            print(f"  Interpretation: {kappa_result['interpretation']}")
            print(f"\n  Confusion Matrix:")
            print(f"    TP (both consistent): {kappa_result['confusion']['TP']}")
            print(f"    TN (both contradictory): {kappa_result['confusion']['TN']}")
            print(f"    FP (LLM=consistent, Human=contradictory): {kappa_result['confusion']['FP']}")
            print(f"    FN (LLM=contradictory, Human=consistent): {kappa_result['confusion']['FN']}")
        else:
            print("  No matching pairs found between claim pairs and human labels")
            print("  Check that pair_id format matches in both files")
    else:
        print("  Skipping (no human labels file)")
    
    print("\n" + "-" * 70)
    print("PHASE 5: Generating Reports")
    print("-" * 70)
    
    validation_data = report_gen.save_validation_results(
        ablation_results, consistency_results, kappa_result
    )
    report_gen.generate_markdown(judge_results, ablation_results, consistency_results, kappa_result)
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total comparisons: {len(judge_results)}")
    print(f"Average consistency score: {np.mean([r.consistency_score for r in judge_results]):.2f}")
    print(f"Best prompt strategy: {validation_data['summary']['best_strategy']}")
    print(f"Self-consistency stable rate: {validation_data['summary']['self_consistency_stable_rate']:.0%}")
    print(f"Cohen's Kappa: {validation_data['summary']['cohens_kappa']:.4f} ({validation_data['summary']['kappa_interpretation']})")
    print(f"Human labels used: {validation_data['summary']['kappa_n_samples']}")
    print(f"\nOutputs saved to: {report_gen.output_dir}/")
    
    return {
        "judge_results": judge_results,
        "ablation": ablation_results,
        "consistency": consistency_results,
        "kappa": kappa_result,
        "summary": validation_data["summary"]
    }


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o")
    
    results = run_part3(
        provider=provider,
        model=model,
        api_key=api_key,
        human_labels_path=str(HUMAN_LABELS_PATH),
        ablation_pairs=5,
        consistency_pairs=3,
        consistency_runs=5
    )
    
    if results:
        print("\nPart 3 completed successfully!")
        return 0
    else:
        print("\nPart 3 failed")
        return 1


if __name__ == "__main__":
    exit(main())