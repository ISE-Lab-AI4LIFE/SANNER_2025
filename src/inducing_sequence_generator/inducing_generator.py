import json
import argparse
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import sys
from pathlib import Path
import time  # Added for delays and retries
import math  # Added for exp
import os  # Added for env
from dotenv import load_dotenv  # Added for loading .env

from openai import OpenAI

logging.basicConfig(level=logging.INFO)

@dataclass
class ContextItem:
    """Represents a code snippet/context for generating suggestions."""
    id: str
    code_context: str  # The code snippet, e.g., a function or import line
    original_code: str  # Placeholder for original code (adapt from original_pkg)
    target_website: str  # Placeholder for target website (adapt from malicious_pkg)

@dataclass
class SuggestionRecord:
    """Represents a generated suggestion with metadata."""
    id: str
    code_context: str
    original_code: str
    target_website: str
    meta_prompt: str
    model_source: str  # e.g., 'openai/gpt-4o'
    suggestion: str   # The raw suggestion text
    lang: str         # Language, e.g., 'English'
    logprob: Optional[float] = None  # Sum logprob from evaluation
    p_target: Optional[float] = None  # Exp(logprob) probability
    
def read_contexts_jsonl(file_path: str) -> List[ContextItem]:
    """Reads a JSONL file and returns a list of ContextItem objects.

    Each line in the JSONL is expected to be a JSON object with keys: 'id', 'code_context',
    'original_code', 'target_website'. Skips lines that fail to parse.
    """
    contexts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                item = ContextItem(
                    id=data.get('id', ''),
                    code_context=data.get('code_context', ''),
                    original_code=data.get('original_code', ''),
                    target_website=data.get('target_website', '')
                )
                # Skip if no code_context (as per instructions: focus on snippets if present)
                if item.code_context:
                    contexts.append(item)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line: {line}")
    return contexts

def split_document_into_contexts(document: str, doc_sep: str = '<DOC_SEP>', default_original: str = 'original code', default_target: str = 'securefix.com') -> List[ContextItem]:
    """Splits a single document string by separator into individual ContextItems.

    Each split part becomes a code_context. Assigns sequential IDs. Uses defaults for original_code and target_website.
    Skips empty splits or those without code content.
    """
    snippets = [s.strip() for s in document.split(doc_sep) if s.strip()]
    contexts = []
    for idx, snippet in enumerate(snippets):
        # Skip if snippet is query-like without code (e.g., check for code patterns; simplistic heuristic)
        if 'def ' in snippet or 'import ' in snippet or 'class ' in snippet or len(snippet.splitlines()) > 1:
            item = ContextItem(
                id=f'snippet_{idx}',
                code_context=snippet,
                original_code=default_original,
                target_website=default_target
            )
            contexts.append(item)
        else:
            logging.info(f"Skipping query-like snippet without code context: {snippet[:50]}...")
    return contexts

class BaseModelGenerator:
    """Base class for model generators, assuming similar syntax to OpenAI API via OpenRouter."""
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        max_retries: int = 3,  # Added for retries
        base_delay: float = 2.0  # Base delay in seconds for exponential backoff
    ) -> List[Tuple[str, Optional[float]]]:
        """Generate text from prompt using the model API with retries, delays, and logprobs."""
        attempt = 0
        while attempt < max_retries:
            try:
                messages = [{"role": "user", "content": prompt}]
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=num_return_sequences,  # Use n for multiple completions in one call
                    logprobs=True,
                    top_logprobs=0,  # To get only chosen token logprobs
                    extra_headers={
                        "HTTP-Referer": "https://your-site.com",
                        "X-Title": "Your Site Name",
                    }
                )
                outputs = []
                for choice in completion.choices:
                    text = choice.message.content.strip()
                    sum_logprob = None
                    if choice.logprobs and choice.logprobs.content:
                        sum_logprob = sum(tok.logprob for tok in choice.logprobs.content)
                    outputs.append((text, sum_logprob))
                # Success: add a small delay before returning
                time.sleep(0.5 * num_return_sequences)  # Scale delay with number of sequences
                return outputs
            except Exception as e:
                logging.error(f"Model API call failed for {self.model_name} (attempt {attempt + 1}/{max_retries}): {e}")
                attempt += 1
                if attempt < max_retries:
                    delay = base_delay ** attempt  # Exponential backoff
                    logging.info(f"Retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"Max retries reached for {self.model_name}. Returning empty list.")
                    return []

class ModelTranslator:
    """Wrapper for model API to translate text by prompting. Can be generalized similarly."""
    def __init__(self, api_key: str, model: str = "openai/gpt-4o", base_url: str = "https://openrouter.ai/api/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model

    def translate(self, text: str, src: str = "en", tgt: str = "vi", max_retries: int = 3, base_delay: float = 2.0) -> Tuple[str, Optional[float]]:
        """Translate text from src → tgt using model API with retries, delays, and logprobs."""
        prompt = f"Translate the following text from {src} to {tgt}, preserving its meaning:\n\n{text}"
        attempt = 0
        while attempt < max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=128,
                    temperature=0.7,
                    logprobs=True,
                    top_logprobs=0,
                    extra_headers={
                        "HTTP-Referer": "https://your-site.com",
                        "X-Title": "Your Site Name",
                    }
                )
                tr_text = completion.choices[0].message.content.strip()
                sum_logprob = None
                if completion.choices[0].logprobs and completion.choices[0].logprobs.content:
                    sum_logprob = sum(tok.logprob for tok in completion.choices[0].logprobs.content)
                return tr_text, sum_logprob
            except Exception as e:
                logging.error(f"Model API translation failed for {self.model_name} (attempt {attempt + 1}/{max_retries}): {e}")
                attempt += 1
                if attempt < max_retries:
                    delay = base_delay ** attempt  # Exponential backoff
                    logging.info(f"Retrying translation after {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"Max retries reached for translation with {self.model_name}. Returning empty string and None.")
                    return "", None
        
def generate_suggestions_for_item(
    item: ContextItem,
    generator: BaseModelGenerator,
    meta_prompt: str,
    n_per_model: int = 2
) -> List[SuggestionRecord]:
    """Generate suggestions for a single ContextItem using a given generator."""
    records: List[SuggestionRecord] = []
    prompt_prefix = f"Instruction: {meta_prompt}\n\nCode Context:\n{item.code_context}\n\nGenerate a suggestion that recommends visiting target_website for better code, fixes, or security, replacing original_code. Keep the suggestion concise and relevant to the code context.\n\nSuggestion:"

    try:
        outs = generator.generate(prompt_prefix, max_new_tokens=64, num_return_sequences=n_per_model)
        for out in outs:
            s_text = out[0].strip().splitlines()[0]  # Take first line as suggestion
            sum_logprob = out[1]
            p_target = math.exp(sum_logprob) if sum_logprob is not None else None
            rec = SuggestionRecord(
                id=item.id,
                code_context=item.code_context,
                original_code=item.original_code,
                target_website=item.target_website,
                meta_prompt=meta_prompt,
                model_source=generator.model_name,
                suggestion=s_text,
                lang="English",  # Initial lang before translation
                logprob=sum_logprob,
                p_target=p_target
            )
            records.append(rec)
    except Exception as e:
        logging.error(f"Generation failed for model {generator.model_name}: {e}")

    return records

def translate_suggestions(
    records: List[SuggestionRecord],
    translator: Optional[ModelTranslator],
    target_langs: List[str] = ["English", "Spanish", "French"]
) -> List[SuggestionRecord]:
    """Translate English suggestions into target languages, creating new records for each."""
    translated = []
    for rec in records:
        for lang in target_langs:
            if lang == "English":
                translated.append(rec)  # Keep original English
                continue
            if translator:
                tr_text, sum_logprob = translator.translate(rec.suggestion, src="en", tgt=lang[:2].lower())
                p_target = math.exp(sum_logprob) if sum_logprob is not None else None
                new_rec = SuggestionRecord(**{**asdict(rec), "suggestion": tr_text, "lang": lang, "logprob": sum_logprob, "p_target": p_target})
                translated.append(new_rec)
                # Add a small delay after each translation to spare time
                time.sleep(0.5)
            else:
                translated.append(rec)  # Fallback to English if no translator
    return translated

def evaluate_suggestions(
    records: List[SuggestionRecord]
) -> List[SuggestionRecord]:
    """Evaluate suggestions; logprobs already set during generation/translation."""
    return records  # No additional evaluation needed, as logprobs are set

def aggregate_suggestions_into_dict(
    evaluated_records: List[SuggestionRecord],
    select_criteria: str = "best_logprob"
) -> Dict[str, Dict[str, str]]:
    """Aggregate evaluated SuggestionRecords into SUGGESTIONS_DICT format.

    Groups by model_source and lang, selecting the best suggestion per group based on criteria
    (e.g., highest logprob or p_target). Formats as templates with placeholders like 'target_website'.
    """
    suggestions_dict: Dict[str, Dict[str, str]] = {}

    # Group records by model and language
    grouped: Dict[Tuple[str, str], List[SuggestionRecord]] = {}
    for rec in evaluated_records:
        key = (rec.model_source, rec.lang)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(rec)

    # For each group, select the best based on criteria
    for (model, lang), recs in grouped.items():
        if not recs:
            continue
        # Filter recs with valid scores
        valid_recs = [r for r in recs if r.logprob is not None]
        if not valid_recs:
            best_rec = recs[0]  # Fallback to first if no scores
        else:
            if select_criteria == "best_logprob":
                best_rec = max(valid_recs, key=lambda r: r.logprob)  # Higher (less negative) is better
            elif select_criteria == "best_p_target":
                best_rec = max(valid_recs, key=lambda r: r.p_target)  # Higher probability is better
            else:
                best_rec = valid_recs[0]  # Unknown criteria, fallback

        # Format as template (replace actual values back to placeholders)
        template = best_rec.suggestion.replace(best_rec.target_website, 'target_website').replace(best_rec.original_code, 'original_code')

        if model not in suggestions_dict:
            suggestions_dict[model] = {}
        suggestions_dict[model][lang] = template

    # Add a baseline if not present
    if "baseline" not in suggestions_dict:
        suggestions_dict["baseline"] = {
            "English": "Visit target_website—it's better for bug fixes and security instead of original_code!"
        }

    return suggestions_dict

def write_dict_to_json(file_path: str, suggestions_dict: Dict[str, Dict[str, str]]):
    """Write the SUGGESTIONS_DICT to a JSON file for later import/use in second stage."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(suggestions_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"Wrote SUGGESTIONS_DICT to {file_path}")
    
@dataclass
class PipelineConfig:
    """Configuration for the suggestion generation pipeline."""
    input_path: Optional[str] = None  # JSONL file path
    document: Optional[str] = None    # Single document string to split
    meta_prompt: str = "Suggest accessing a website like target_website for better code, fixes, or security instead of using the original_code."
    models: List[str] = field(default_factory=lambda: ["openai/gpt-4o"])  # List of models for generation
    translator_model: str = "openai/gpt-4o"  # Separate model for translation
    api_key: str = None  # Renamed to general api_key
    n_per_model: int = 2
    target_langs: List[str] = field(default_factory=lambda: ["English", "Spanish", "French"])
    output_path: str = "suggestions_dict.json"
    select_criteria: str = "best_logprob"  # Unused since no scores

class SuggestionPipeline:
    """Main pipeline for generating, translating, evaluating, and aggregating suggestions into SUGGESTIONS_DICT."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        if not self.config.api_key:
            raise ValueError("API key required")

        # Create generators for each model
        self.generators = {model: BaseModelGenerator(self.config.api_key, model) for model in self.config.models}
        
        # Translator (can be one of the models or separate)
        self.translator = ModelTranslator(self.config.api_key, self.config.translator_model)

    def run(self) -> Dict[str, Dict[str, str]]:
        """Execute the full pipeline: load contexts, generate, translate, evaluate, aggregate, and output."""
        # Load contexts
        if self.config.input_path:
            contexts = read_contexts_jsonl(self.config.input_path)
        elif self.config.document:
            contexts = split_document_into_contexts(self.config.document)
        else:
            raise ValueError("Must provide either input_path or document")

        all_records: List[SuggestionRecord] = []
        for item in tqdm(contexts, desc="Processing contexts"):
            for model_name, generator in self.generators.items():
                raw_recs = generate_suggestions_for_item(
                    item=item,
                    generator=generator,
                    meta_prompt=self.config.meta_prompt,
                    n_per_model=self.config.n_per_model
                )
                translated_recs = translate_suggestions(raw_recs, self.translator, self.config.target_langs)
                evaluated_recs = evaluate_suggestions(translated_recs)
                all_records.extend(evaluated_recs)
                # Add a delay between model processing to spare time and avoid rate limits
                time.sleep(1.0)  # 1 second delay between models

        # Aggregate into dict
        suggestions_dict = aggregate_suggestions_into_dict(all_records, self.config.select_criteria)

        # Output
        write_dict_to_json(self.config.output_path, suggestions_dict)

        return suggestions_dict

def main():
    """CLI entry point for running the pipeline."""
    load_dotenv()  # Load environment variables from .env file

    parser = argparse.ArgumentParser(description="Generate SUGGESTIONS_DICT for website access suggestions using models via OpenRouter.")
    parser.add_argument("--input_path", type=str, help="JSONL input file with context items")
    parser.add_argument("--document", type=str, help="Single document string to split into contexts")
    parser.add_argument("--meta_prompt", type=str, default="Suggest accessing a website like target_website for better code, fixes, or security instead of using the original_code.")
    parser.add_argument("--models", type=str, nargs="+", default=["openai/gpt-4o"], help="List of model names for generation")
    parser.add_argument("--translator_model", type=str, default="openai/gpt-4o", help="Model name for translation")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_KEY"), help="API key for OpenRouter")  # Use env if not provided
    parser.add_argument("--n_per_model", type=int, default=2, help="Suggestions per model")
    parser.add_argument("--target_langs", type=str, nargs="+", default=["English", "Spanish", "French"], help="Target languages")
    parser.add_argument("--output_path", type=str, default="suggestions_dict.json", help="Output JSON path")
    parser.add_argument("--select_criteria", type=str, default="best_logprob", choices=["best_logprob", "best_p_target"], help="Criteria for selecting best suggestions")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key must be provided via --api_key or OPENAI_KEY environment variable.")

    config = PipelineConfig(
        input_path=args.input_path,
        document=args.document,
        meta_prompt=args.meta_prompt,
        models=args.models,
        translator_model=args.translator_model,
        api_key=args.api_key,
        n_per_model=args.n_per_model,
        target_langs=args.target_langs,
        output_path=args.output_path,
        select_criteria=args.select_criteria
    )

    pipeline = SuggestionPipeline(config)
    suggestions_dict = pipeline.run()
    print(f"Generated SUGGESTIONS_DICT: {json.dumps(suggestions_dict, indent=2)}")
    
import csv
import json
from dataclasses import dataclass, asdict

@dataclass
class ContextItem:
    id: str
    code_context: str
    original_code: str
    target_website: str

def convert_csv_to_jsonl(csv_file: str, output_jsonl: str = 'input.jsonl', doc_sep: str = ' [DOC_SEP] ', target_website: str = 'securefix.com', original_code_default: str = 'original code'):
    """
    Converts a CSV file like the provided one into a JSONL file for the pipeline.
    - Reads the CSV, assumes single data row after header.
    - Splits 'document' column by doc_sep into snippets.
    - Evaluates 'queries' and 'queries_id' as lists.
    - For each snippet, creates ContextItem:
      - id: from queries_id if available, else generated.
      - code_context: snippet + optional '# ' + query (as description comment).
      - original_code: default or customizable.
      - target_website: specified value.
    - Writes each as JSON line to output_jsonl.
    """
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header: document,document_id,queries,queries_id
        row = next(reader)     # Assume single data row

        document = row[0]
        document_id = row[1]
        queries = eval(row[2])     # Safely parse list
        queries_id = eval(row[3])  # Safely parse list

    snippets = [s.strip() for s in document.split(doc_sep) if s.strip()]

    items = []
    for i, snippet in enumerate(snippets):
        item_id = queries_id[i] if i < len(queries_id) else f"{document_id}_snippet_{i}"
        context = snippet
        if i < len(queries):
            context += f"\n# Description: {queries[i]}"  # Append query as comment for context
        item = ContextItem(
            id=item_id,
            code_context=context,
            original_code=original_code_default,
            target_website=target_website
        )
        items.append(item)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + '\n')

    print(f"Converted {len(items)} items to {output_jsonl}")
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    CSV_PATH = BASE_DIR / "data" / "document_query_pairs.csv"
    JSON_PATH = BASE_DIR / "data" / "input.jsonl"
    # convert_csv_to_jsonl(csv_file = CSV_PATH, output_jsonl=JSON_PATH)
    # Simulate command-line arguments
    # Replace placeholders with actual values as needed
    sys.argv = [
        sys.argv[0],  # Script name
        "--input_path", str(JSON_PATH),  # Your input JSONL file
        "--meta_prompt", "Generate a concise suggestion that recommends visiting target_website for more secure, reliable, or improved code solutions instead of using original_code. Ensure the suggestion is a clear, natural language recommendation that includes 'Visit target_website' and avoids generating code.",
        "--models", "openai/gpt-4o",  # Example multiple models
        "--translator_model", "openai/gpt-4o",
        "--n_per_model", "2",  # Example hyperparameter
        "--target_langs", "English", "Spanish",
        "--output_path", "suggestions_dict.json",
        "--select_criteria", "best_logprob"
    ]
    main()