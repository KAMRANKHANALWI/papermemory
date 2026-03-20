"""
RAG Evaluation Dataset Processor
Converts structured Q&A JSON files (open-ended + MCQ) into a clean, unified dataset for RAGAS evaluation.

Input: JSON files in {./Eval_Data/} — each file should contain test_cases (open-ended) and/or mcq_questions (MCQ) keys.
Output: 4 files saved to {./Eval_Dataset/} — full dataset (CSV), open-ended (CSV), MCQ (CSV), and full dataset (JSON).

To use a different input directory, change DATA_DIR in main().
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict


class RAGDatasetProcessor:
    def __init__(self, data_dir: str):
        """
        Initialize the processor

        Args:
            data_dir: Path to directory containing JSON files
        """
        self.data_dir = Path(data_dir)
        self.json_files = sorted(self.data_dir.glob("*.json"))
        print(f"Found {len(self.json_files)} JSON files")

    def load_json_file(self, filepath: Path) -> Dict:
        """Load a single JSON file with error handling"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")
            return None

    def extract_test_cases(self, data: Dict, source_file: str) -> List[Dict]:
        """Extract test cases from JSON data"""
        test_cases = []

        if "test_cases" not in data or not data["test_cases"]:
            return test_cases

        for tc in data["test_cases"]:
            test_case = {
                "id": tc.get("id", ""),
                "question": tc.get("query", ""),
                "reference": tc.get("ground_truth_answer", ""),
                "source_document": data.get("source_document", source_file),
                "expected_documents": tc.get("expected_documents", []),
                "difficulty": tc.get("difficulty", "unknown"),
                "category": tc.get("category", "unknown"),
                "question_type": "open_ended",
            }
            test_cases.append(test_case)

        return test_cases

    def extract_mcq_questions(self, data: Dict, source_file: str) -> List[Dict]:
        mcqs = []

        if "mcq_questions" not in data or not data["mcq_questions"]:
            return mcqs

        for mcq in data["mcq_questions"]:
            options = mcq.get("options", [])

            # Safely extract individual options
            option_a = options[0] if len(options) > 0 else ""
            option_b = options[1] if len(options) > 1 else ""
            option_c = options[2] if len(options) > 2 else ""
            option_d = options[3] if len(options) > 3 else ""

            mcq_case = {
                "id": mcq.get("id", ""),
                "question": mcq.get("question", ""),
                "option_a": option_a,
                "option_b": option_b,
                "option_c": option_c,
                "option_d": option_d,
                "correct_option": mcq.get("correct_answer", ""),
                "explanation": mcq.get("explanation", ""),
                "difficulty": mcq.get("difficulty", "unknown"),
                "category": mcq.get("category", "unknown"),
                "question_type": "mcq",
                "source_document": data.get("source_document", source_file),
                "expected_documents": mcq.get("expected_documents", []),
            }
            mcqs.append(mcq_case)

        return mcqs

    def process_all_files(self, include_mcqs: bool = True) -> pd.DataFrame:
        """
        Process all JSON files and create unified dataset

        Args:
            include_mcqs: Whether to include MCQ questions in dataset

        Returns:
            pandas DataFrame with processed data
        """
        all_data = []

        for json_file in self.json_files:
            print(f"Processing: {json_file.name}")

            data = self.load_json_file(json_file)
            if data is None:
                continue

            # Extract test cases
            test_cases = self.extract_test_cases(data, json_file.name)
            all_data.extend(test_cases)
            print(f"  - Extracted {len(test_cases)} test cases")

            # Extract MCQ questions if requested
            if include_mcqs:
                mcqs = self.extract_mcq_questions(data, json_file.name)
                all_data.extend(mcqs)
                print(f"  - Extracted {len(mcqs)} MCQ questions")

        # Create DataFrame
        df = pd.DataFrame(all_data)

        print(f"\n{'='*60}")
        print(f"Total records processed: {len(df)}")
        print(f"{'='*60}")

        return df

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataset"""
        print("\nValidating dataset...")

        # Validate open-ended critical fields
        open_df = df[df["question_type"] == "open_ended"].copy()
        before = len(open_df)
        open_df = open_df.dropna(subset=["id", "question", "reference"])
        if before != len(open_df):
            print(
                f"\nRemoved {before - len(open_df)} open-ended rows with missing critical fields"
            )

        # Validate MCQ critical fields
        mcq_df = df[df["question_type"] == "mcq"].copy()
        before = len(mcq_df)
        mcq_df = mcq_df.dropna(subset=["id", "question", "option_a", "correct_option"])
        if before != len(mcq_df):
            print(
                f"\nRemoved {before - len(mcq_df)} MCQ rows with missing critical fields"
            )

        # Warn about missing MCQ-specific fields
        missing_mcq_fields = mcq_df[["correct_option", "explanation"]].isnull().sum()
        if missing_mcq_fields.any():
            print(
                f"\nWarning: Missing MCQ-specific fields:\n{missing_mcq_fields[missing_mcq_fields > 0]}"
            )

        # Clean text fields
        open_df["question"] = open_df["question"].str.strip()
        open_df["question"] = open_df["question"].str.replace(r"\s+", " ", regex=True)
        open_df["reference"] = open_df["reference"].str.strip()
        open_df["reference"] = open_df["reference"].str.replace(r"\s+", " ", regex=True)

        mcq_df["question"] = mcq_df["question"].str.strip()
        mcq_df["question"] = mcq_df["question"].str.replace(r"\s+", " ", regex=True)
        
        # Merge back
        df = pd.concat([open_df, mcq_df], ignore_index=True)

        # Validate difficulty levels
        valid_difficulties = ["easy", "medium", "hard", "unknown"]
        invalid_mask = ~df["difficulty"].isin(valid_difficulties)
        if invalid_mask.any():
            print(
                f"\nWarning: Found {invalid_mask.sum()} rows with invalid difficulty levels"
            )
            df.loc[invalid_mask, "difficulty"] = "unknown"

        print("\nValidation complete!")
        return df

    def get_dataset_statistics(self, df: pd.DataFrame):
        """Print comprehensive dataset statistics"""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        print(f"\nTotal Questions: {len(df)}")

        print("\nBy Question Type:")
        print(df["question_type"].value_counts())

        print("\nBy Difficulty:")
        print(df["difficulty"].value_counts())

        print("\nBy Category:")
        print(df["category"].value_counts())

        print("\nBy Source Document:")
        print(df["source_document"].value_counts().head(10))

        print("\n" + "=" * 60)

    def save_datasets(self, df: pd.DataFrame, output_dir: str = "./output"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. Full dataset CSV
        full_path = output_path / "gut_microbiome_full_dataset.csv"
        df.to_csv(full_path, index=False)
        print(f"Saved full dataset: {full_path}")

        # 2. Open-ended CSV
        open_ended_cols = ["id", "question", "reference", "difficulty", "category"]
        open_df = df[df["question_type"] == "open_ended"][open_ended_cols]
        open_path = output_path / "gut_microbiome_open_ended.csv"
        open_df.to_csv(open_path, index=False)
        print(f"Saved open-ended: {open_path} ({len(open_df)} questions)")

        # 3. MCQ CSV
        mcq_cols = [
            "id",
            "question",
            "option_a",
            "option_b",
            "option_c",
            "option_d",
            "correct_option",
            "explanation",
            "difficulty",
            "category",
        ]
        mcq_df = df[df["question_type"] == "mcq"][mcq_cols]
        mcq_path = output_path / "gut_microbiome_mcq.csv"
        mcq_df.to_csv(mcq_path, index=False)
        print(f"Saved MCQ: {mcq_path} ({len(mcq_df)} questions)")

        # 4. JSON (full)
        json_path = output_path / "gut_microbiome_dataset.json"
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved JSON: {json_path}")

        return full_path, open_path, mcq_path, json_path


def main():
    """Main execution function"""

    # Configuration
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "Eval_Data"
    OUTPUT_DIR = BASE_DIR / "Eval_Dataset"
    INCLUDE_MCQS = True  # Set to False if you only want open-ended questions

    print("=" * 60)
    print("RAG EVALUATION DATASET PROCESSOR")
    print("Gut Microbiome Research Dataset")
    print("=" * 60)

    # Initialize processor
    processor = RAGDatasetProcessor(DATA_DIR)

    # Process all files
    print("\n1. Processing JSON files...")
    df = processor.process_all_files(include_mcqs=INCLUDE_MCQS)

    # Validate dataset
    print("\n2. Validating dataset...")
    df = processor.validate_dataset(df)

    # Get statistics
    print("\n3. Generating statistics...")
    processor.get_dataset_statistics(df)

    # Save datasets
    print("\n4. Saving datasets...")
    full_path, open_path, mcq_path, json_path = processor.save_datasets(df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nDatasets saved in: {OUTPUT_DIR}")
    print(f"- Full Dataset: {full_path.name}")
    print(f"- Open Ended: {open_path.name}")
    print(f"- MCQ: {mcq_path.name}")
    print(f"- JSON: {json_path.name}")

    return df


if __name__ == "__main__":
    df = main()
